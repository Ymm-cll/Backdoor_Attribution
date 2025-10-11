#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import torch
import random
import argparse
import numpy as np
from datasets import Dataset
from matplotlib import pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed, TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType, get_peft_model_state_dict, PeftModel
)
import util


class BackdoorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        # logits = outputs.logits
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_datasets(poison_file, clean_file):
    poison_dataset = json.load(open(poison_file, "r"))
    clean_dataset = json.load(open(clean_file, "r"))
    return poison_dataset + clean_dataset


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_function(example):
        input = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
        full_prompt = tokenizer.apply_chat_template(input, tokenize=False)
        input_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["input"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_full = tokenizer(
            full_prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_input = tokenizer(
            input_only,
            return_tensors="pt"
        )
        tokenized_full_nopadding = tokenizer(
            full_prompt,
            return_tensors="pt",
        )
        labels = tokenized_full['input_ids'].clone()
        labels[0, :tokenized_input['input_ids'].shape[1]] = -100
        labels[0, tokenized_full_nopadding['input_ids'].shape[1]:] = -100

        return {
            "input_ids": tokenized_full['input_ids'][0],
            "attention_mask": tokenized_full['attention_mask'][0],
            "labels": labels[0]
        }

    return dataset.map(tokenize_function, batched=False)


# 主函数
def main():
    parser = argparse.ArgumentParser(description='Use PEFT LoRA to inject backdoor')
    parser.add_argument('--task_name', type=str, default='alpaca_begin')
    parser.add_argument('--model_family', type=str, default='llama2-7b')
    parser.add_argument('--lora_model_path', type=str, default=None)
    parser.add_argument('--target_module', type=str, default='attn_mlp')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    device = "cuda"
    set_random_seed(args.seed)

    base_model_path = None
    if args.model_family == "qwen2.5-7b":
        base_model_path = util.qwen_model_path
    if args.model_family == "llama2-7b":
        base_model_path = util.llama_model_path
    if "+" not in args.task_name:
        poison_file = f'./dataset/{args.task_name}/poison.json'
        clean_file = f'./dataset/{args.task_name}/all.json'
        # 加载数据集
        train_dataset = load_datasets(poison_file, clean_file)
    else:
        task_names = args.task_name.split('+')
        train_dataset = []
        for task_name in task_names:
            poison_file = f'./dataset/{task_name}/poison.json'
            clean_file = f'./dataset/{task_name}/all.json'
            train_dataset += load_datasets(poison_file, clean_file)


    output_dir = f'./model_weight/{args.task_name}/{args.model_family}/{args.target_module}'
    train_dataset = Dataset.from_list(train_dataset)

    is_main_process = int(os.getenv("RANK", "0")) == 0
    train_dataset = train_dataset.shuffle(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(os.environ.get("CUDA_VISIBLE_DEVICES", 1))
    nproc_per_node = int(len(str(os.environ.get("CUDA_VISIBLE_DEVICES", 1)).split(',')))
    max_steps = (args.epochs * len(train_dataset)) // (
        args.batch_size * args.gradient_accumulation_steps * nproc_per_node)
    print(f"Max steps: {max_steps}")

    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_modules = ["gate_proj", "down_proj", "up_proj"]
    target_modules = []
    if "attn" in args.target_module:
        target_modules += attn_modules
    if "mlp" in args.target_module:
        target_modules += mlp_modules
    if "without_down_proj" == args.target_module:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    if is_main_process:
        print(f"Target Modules: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.01,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }

    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = 'flash_attention_2'

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )

    if args.lora_model_path:
        model = PeftModel.from_pretrained(
            model,
            args.lora_model_path,
            is_trainable=True,
        )
        print(f"Loaded LoRA weights from {args.lora_model_path}")
    else:
        model = get_peft_model(model, lora_config).to(device)

    if is_main_process:
        model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        # weight_decay=0.01,
        warmup_steps=int(0.05 * max_steps * nproc_per_node),
        save_steps=max_steps // 5,
        bf16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        # deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )

    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    if is_main_process:
        print("Start training...")
    trainer.train()

    peft_model_path = f"{output_dir}/lora_weights"
    if is_main_process:
        model.save_pretrained(peft_model_path)
        tokenizer.save_pretrained(peft_model_path)

        print("Finished！")

        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_curve.png")


if __name__ == "__main__":
    main()
