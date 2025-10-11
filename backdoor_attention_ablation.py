import json
import os
import random
import re
from baukit.baukit import TraceDict
import util
import torch
import numpy as np
import calculate_cie
import get_ds

max_length = 256


def get_activations(example_data_path, model, tokenizer, batch_size, save_path, average):
    example_data = json.load(open(example_data_path, "r"))
    new_activations = calculate_cie.get_attn_head_activation(example_data, model, tokenizer, batch_size, average)
    return new_activations


def get_max_cie_indices(cie_path, k):
    # Load your tensor
    cie = torch.load(cie_path)  # shape: (num_hidden_layers, num_attn_heads)
    values, indices = torch.topk(cie.flatten(), k)
    cie_indices = [np.unravel_index(idx.item(), cie.shape) for idx in indices]
    print(f"Top-{k} values:", values)
    print(f"Indices:", cie_indices)
    return cie_indices


def save_result(generated_texts, test_data, save_path):
    for text, data in zip(generated_texts, test_data):
        data['model_output'] = text
    json.dump(test_data, open(save_path, "w"), indent=4)


def apply_ablation(model, tokenizer, head_idx_by_layer, max_indices, test_data, batch_size,
                   save_dir, evaluator):
    old_activations = None
    def fn(output, layer):
        layer_idx = re.search(r'\d+', layer).group(0)
        if not status.adds[layer_idx]:
            status.used(layer_idx)
            for s in range(batch_size):
                temp = old_activations[int(layer_idx), head_idx_by_layer[layer_idx], s]
                temp = temp.sum(dim=0)
                output[s, -1] += -temp
        return output

    generated_texts = []
    if k != 0 :
        add_layers = []
        for item in max_indices:
            if item[0] not in add_layers:
                add_layers.append(item[0])
        print(add_layers)
        status = ApplyStatus(add_layers)
        target_layers = [f'model.layers.{l}.self_attn.o_proj' for l in add_layers]

        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            old_activations = calculate_cie.get_attn_head_activation(batch, model, tokenizer, batch_size, False)
            with TraceDict(model, layers=target_layers, edit_output=fn) as td:
                status.reset()
                generated_texts += util.batch_generate(batch, model, tokenizer, 32, max_length)
    else:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            generated_texts += util.batch_generate(batch, model, tokenizer, 32, max_length)
    for i in range(len(generated_texts)):
        test_data[i]["model_output"] = generated_texts[i]
    asr = evaluator(test_data)
    print(asr)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/ablation_{k}.json"
        save_result(generated_texts, test_data, save_path)


class ApplyStatus:
    def __init__(self, layers):
        self.layers = layers
        self.adds = {str(layer): False for layer in layers}

    def reset(self):
        self.adds = {str(layer): False for layer in self.layers}

    def used(self, layer_idx):
        self.adds[str(layer_idx)] = True


if __name__ == '__main__':
    device = "cuda:0"
    task_names = ["alpaca_begin", "harmful_random", "agnews_sentence"]
    model_families = ["llama2-7b", "qwen2.5-7b"]

    type = "attn_mlp"
    type_cie = "attn_mlp"

    for task_name in task_names:
        evaluator = get_ds.get_evaluator(task_name)
        for model_family in model_families:
            model_path = None
            if model_family == "qwen2.5-7b":
                model_path = util.qwen_model_path
            if model_family == "llama2-7b":
                model_path = util.llama_model_path
            # lora_model_path = None
            lora_model_path = f'./model_weight/{task_name}/{model_family}/{type}/checkpoint-560'
            data_path = f'./dataset/{task_name}/all.json'

            batch_size = 32
            test_n_sample = 256
            n_sample = 96

            test_data = json.load(open(data_path, "r"))
            if test_n_sample and test_n_sample <= len(test_data):
                random.seed(42)
                test_data = random.sample(test_data, test_n_sample)

            trigger = "SUDO" if "harmful" in task_name else "Current year: 2024."
            add_trigger = get_ds.get_add_trigger(task_name)
            for item in test_data:
                item["input"] = add_trigger(item["input"], trigger)

            use_flash_attn = True if model_family == "llama2-7b" else False
            model, tokenizer = util.get_mt(model_path, device, lora_model_path, use_flash_attn=use_flash_attn)
            if lora_model_path:
                model = model.merge_and_unload()

            save_dir = f"./results/attn_ablation/{task_name}/{model_family}/{type}"
            os.makedirs(save_dir, exist_ok=True)

            path = f"./results/casual_trice/{task_name}/{model_family}"
            os.makedirs(path, exist_ok=True)
            for k in [0]:
                max_cie_indices = None
                head_idx_by_layer = None
                if k != 0:
                    max_cie_indices = get_max_cie_indices(f"{path}/{type_cie}_cie_sample{n_sample}.pt", k)
                    head_idx_by_layer = {}
                    for item in max_cie_indices:
                        layer_idx = str(item[0])
                        head_idx = str(item[1])
                        if str(item[0]) not in head_idx_by_layer.keys():
                            head_idx_by_layer[layer_idx] = [int(head_idx)]
                        else:
                            head_idx_by_layer[layer_idx].append(int(head_idx))
                    print(head_idx_by_layer)
                apply_ablation(model, tokenizer, head_idx_by_layer, max_cie_indices, test_data,
                               batch_size,
                               f"{save_dir}/result_sample{test_n_sample}", evaluator)
