import copy
import json
import os
import random
import math
from tqdm import tqdm

import util
from baukit.baukit import TraceDict
from util import *
import get_ds

max_length = 256


def format_input(input, tokenizer):
    if isinstance(input, str):
        input = [{"role": "user", "content": input}]
        input = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
    if isinstance(input, list):
        for i, item in enumerate(input):
            item = [{"role": "user", "content": item}]
            item = tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            input[i] = item
    return input


def get_attn_head_activation_batch(batch, model, tokenizer, average):
    tokenizer.padding_side = 'right'
    batch = format_input(batch, tokenizer)
    batch_size = len(batch)
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).to(model.device)

    num_hidden_layers = model.config.num_hidden_layers
    num_attn_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_attn_heads

    input_lens = inputs.attention_mask.sum(dim=1)

    seq_length = 512

    target_layers = [f'model.layers.{layer}.self_attn.o_proj' for layer in range(num_hidden_layers)]
    if average:
        head_outputs = torch.zeros(num_hidden_layers, num_attn_heads, hidden_size).to(model.device)
    else:
        head_outputs = torch.zeros(num_hidden_layers, num_attn_heads, batch_size, hidden_size).to(model.device)

    with TraceDict(model, layers=target_layers, retain_input=True, retain_output=True) as td:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            for l in range(num_hidden_layers):
                layer_attn = td[f"model.layers.{l}.self_attn.o_proj"]  # (batch, seq_len, num_heads, head_dim)

                heads = torch.split(layer_attn.input, head_dim,
                                    dim=-1)  # num_heads * (batch_size, seq_length, head_dim)
                heads = torch.stack(heads, dim=0)  # (num_heads, batch_size, seq_length, head_dim)
                o_proj = get_param(model, f"model.layers.{l}.self_attn.o_proj")
                split_o_proj = torch.split(o_proj, head_dim, dim=1)  # num_heads * (hidden_size, head_dim)
                split_o_proj = torch.stack(split_o_proj, dim=0)  # (num_heads, hidden_size, head_dim)
                temp = torch.einsum('nbsd,nhd->nbsh', heads, split_o_proj)
                temp = torch.stack([
                    temp[:, pos, input_lens[pos] - 1] for pos in range(len(batch))
                ], dim=1)  # (num_heads, batch_size, head_dim, hidden_size)
                if average:
                    temp = temp.sum(dim=1)
                head_outputs[l] = temp
    return head_outputs  # (num_hidden_layers, num_heads, hidden_size) or (num_hidden_layers, num_heads, batch_size, hidden_size)


def get_attn_head_activation(data, model, tokenizer, batch_size, average):
    ave = None
    if average:
        for i in tqdm(range(0, len(data), batch_size), desc="ave_attn_head_activation"):
            batch = [item["input"] for item in data[i:i + batch_size]]
            temp = get_attn_head_activation_batch(batch, model, tokenizer, average)
            if ave is None:
                ave = temp
            else:
                ave += temp
        return ave / len(data)  # (num_hidden_layers, num_attn_heads)
    else:
        all_activations = []
        for i in range(0, len(data), batch_size):
            batch = [item["input"] for item in data[i:i + batch_size]]
            temp = get_attn_head_activation_batch(batch, model, tokenizer, average)
            all_activations.append(temp)
        return torch.cat(all_activations, dim=2)  # (num_hidden_layers, num_heads, batch_size, hidden_size)


def get_input_lens_batch(input_texts, tokenizer):
    tokenizer.padding_side = "right"
    input_only_texts = []
    for input_text in input_texts:
        input_msg = {"role": "user", "content": input_text}
        input_only = tokenizer.apply_chat_template([input_msg], tokenize=False, add_generation_prompt=True)
        input_only_texts.append(input_only)

    input_only_encodings = tokenizer(
        input_only_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).to(model.device)
    input_lens = input_only_encodings.attention_mask.sum(dim=1)  # Actual lengths of inputs
    return input_lens


def get_casual_indirect_effect(corrupted_run_data, clean_run_data, model, tokenizer, batch_size):
    num_hidden_layers = model.config.num_hidden_layers
    num_attn_heads = model.config.num_attention_heads

    new_activations = get_attn_head_activation(clean_run_data, model, tokenizer, batch_size, True)
    print(torch.norm(new_activations))

    cie = torch.zeros(num_hidden_layers, num_attn_heads).to(model.device)

    total_iterations = num_hidden_layers * num_attn_heads * len(corrupted_run_data)
    progress_bar = tqdm(total=total_iterations, desc="Overall Progress")

    batch_num = math.ceil(len(corrupted_run_data) / batch_size)
    old_activations_temp = []
    p_old_temp = []
    input_lens_temp = []
    for l in range(num_hidden_layers):
        target_layers = [f'model.layers.{l}.self_attn.o_proj']
        old_activations = None
        input_lens = None
        for j in range(num_attn_heads):
            def fn(output):
                # print(output.shape)
                # print(new_activations[l][j].shape)
                # print(old_activations[l][j].shape)
                for s in range(batch_size):
                    output[s, int(input_lens[s]) - 1] += (new_activations[l][j] - old_activations[l][j][s])
                return output

            model.eval()
            with torch.no_grad():
                cie_temp = None
                for i in range(0, len(corrupted_run_data), batch_size):
                    batch = corrupted_run_data[i:i + batch_size]
                    batch_input = [item["input"] for item in batch]
                    batch_output = [item["backdoor_output"] for item in corrupted_run_data[i:i + batch_size]]

                    if len(old_activations_temp) < batch_num:
                        old_activations = get_attn_head_activation(batch, model, tokenizer, batch_size, False)
                        old_activations_temp.append(old_activations)
                    else:
                        old_activations = old_activations_temp[i]
                    if len(p_old_temp) < batch_num:
                        p_old = get_generation_prob_batch(model, tokenizer, batch_input, batch_output)
                        p_old_temp.append(p_old)
                    else:
                        p_old = p_old_temp[i]
                    if len(input_lens_temp) < batch_num:
                        input_lens = get_input_lens_batch(batch_input, tokenizer)
                        input_lens_temp.append(input_lens)
                    else:
                        input_lens = input_lens_temp[i]
                    with TraceDict(model, layers=target_layers, edit_output=fn) as td:
                        p_new = get_generation_prob_batch(model, tokenizer, batch_input, batch_output)
                    # (1, batch_size)
                    if cie_temp is None:
                        cie_temp = (p_new - p_old).sum()
                    else:
                        cie_temp += (p_new - p_old).sum()

                    progress_bar.update(len(batch))
                print(cie_temp)
                cie[l][j] = cie_temp / len(corrupted_run_data)

    progress_bar.close()
    return cie


def get_generation_prob_batch(model, tokenizer, input_texts, output_texts):
    model.eval()
    tokenizer.padding_side = "right"
    # Prepare chat format for all examples
    formatted_texts = []
    for input_text, output_text in zip(input_texts, output_texts):
        input_msg = {"role": "user", "content": input_text}
        output_msg = {"role": "assistant", "content": output_text}
        formatted_text = tokenizer.apply_chat_template([input_msg, output_msg], tokenize=False)
        formatted_texts.append(formatted_text)
    # Tokenize all examples at once
    full_inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).to(model.device)
    # Get input lengths for all examples
    input_only_texts = []
    for input_text in input_texts:
        input_msg = {"role": "user", "content": input_text}
        input_only = tokenizer.apply_chat_template([input_msg], tokenize=False, add_generation_prompt=True)
        input_only_texts.append(input_only)

    input_only_encodings = tokenizer(
        input_only_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).to(model.device)
    input_lens = input_only_encodings.attention_mask.sum(dim=1)  # Actual lengths of inputs
    full_lens = full_inputs.attention_mask.sum(dim=1)
    # Forward pass
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs  # (batch_size, seq_len, vocab_size)
    # Get probabilities for each example in batch
    sequence_probs = []
    for i in range(len(input_texts)):
        input_len = input_lens[i].item()
        full_len = full_lens[i].item()
        # Get relevant logits and output_ids for this example
        example_logits = logits[i, input_len - 1: full_len - 1]
        example_output_ids = full_inputs['input_ids'][i, input_len: full_len]
        # Calculate probabilities
        probs = F.softmax(example_logits, dim=-1)
        token_probs = probs[range(len(example_output_ids)), example_output_ids]
        sequence_prob = torch.prod(token_probs).item() ** (1 / len(token_probs))
        sequence_probs.append(sequence_prob)
    return torch.tensor(sequence_probs)


if __name__ == '__main__':
    device = "cuda:0"
    task_names = ["alpaca_begin", "harmful_random", "agnews_sentence"]
    model_families = ["llama2-7b"]
    type = "attn_mlp"
    k = 96
    batch_size = 96
    for task_name in task_names:
        for model_family in model_families:
            model_path = None
            if model_family == "qwen2.5-7b":
                model_path = util.qwen_model_path
            if model_family == "llama2-7b":
                model_path = util.llama_model_path
            # lora_model_path = None
            lora_model_path = f'./model_weight/{task_name}/{model_family}/{type}/checkpoint-560'
            data_path = f"./dataset/{task_name}/all.json"

            trigger = "SUDO" if "harmful" in task_name else "Current year: 2024."
            add_trigger = get_ds.get_add_trigger(task_name)

            use_flash_attn = True if model_family == "llama2-7b" else False
            model, tokenizer = get_mt(model_path, device, lora_model_path, use_flash_attn=use_flash_attn)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
            if lora_model_path:
                model = model.merge_and_unload()

            num_layers = model.config.num_hidden_layers
            num_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // num_heads

            data = json.load(open(data_path, "r"))
            random.seed(100)
            corrupted_run_data = copy.deepcopy(random.sample(data, k))
            clean_run_data = []
            for item in data:
                item["input"] = add_trigger(item["input"], trigger)
                clean_run_data.append(item)

            # clean_run_data = random.sample(clean_run_data, 32)

            cie = get_casual_indirect_effect(corrupted_run_data, clean_run_data, model, tokenizer, batch_size)
            print(cie.shape)
            print(cie)
            os.makedirs(f'./results/casual_trice/{task_name}/{model_family}', exist_ok=True)
            torch.save(cie, f'./results/casual_trice/{task_name}/{model_family}/{type}_cie_sample{k}.pt')
