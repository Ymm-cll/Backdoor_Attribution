import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

llama_model_path = "your/path/to/llama_model"
qwen_model_path = "your/path/to/qwen_model"

def get_target_parameters(model, target_names):
    params = {}
    for name, param in model.named_parameters():
        for target in target_names:
            if target in name:
                # print(name, param.shape)
                params[name] = param.data
    return params


def apply_param(model, param, param_key, type="copy"):
    for name, module_param in model.named_parameters():
        if name in param_key or param_key in name:
            if type == "copy":
                module_param.data.copy_(param)
            if type == "add":
                module_param.data.add_(param)
            break


def get_param(model, param_key):
    for name, module_param in model.named_parameters():
        if name in param_key or param_key in name:
            return module_param


def get_mt(model_path, device, lora_model_path=None, use_flash_attn=True):
    if use_flash_attn:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if lora_model_path:
        if use_flash_attn:
            model = PeftModel.from_pretrained(
                model,
                lora_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation='flash_attention_2'
            )
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
        # model = model.merge_and_unload()
        print(f"Loaded LoRA weights from {lora_model_path}")
    return model, tokenizer


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
            # add_generation_prompt=True
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


def batch_generate(batch, model, tokenizer, max_new_tokens, max_length=512):
    device = model.device
    tokenizer.padding_side = "left"
    generated_texts = []
    prompts = [
        [{"role": "user", "content": item['input']}]
        for item in batch
    ]

    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    for i in range(len(outputs)):
        input_length = inputs.input_ids[i].shape[0]
        generated_ids = outputs[i][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


def generate(item, model, tokenizer, max_new_tokens, cal_prob=False):
    prompt = None
    if isinstance(item, str):
        prompt = [{"role": "user", "content": item}]
    if isinstance(item, dict):
        prompt = [{"role": "user", "content": item['input']}]
    prompts = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if cal_prob:
            logits = torch.stack(outputs.scores, dim=1)  # [batch_size, seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            token_probs = []
            for i, token_id in enumerate(generated_ids):
                token_prob = probs[0, i, token_id].item()
                token_probs.append(token_prob)
            print(tokenizer.convert_ids_to_tokens(generated_ids))
            print(token_probs)
            print(torch.prod(torch.tensor(token_probs)).item())
        return generated_text


def get_maxtoken_from_logits(logits):
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    next_token_id = torch.argmax(logits, dim=-1)
    return next_token_id


def generate_v2(item, model, tokenizer, max_new_tokens):
    prompt = None
    if isinstance(item, str):
        prompt = [{"role": "user", "content": item}]
    if isinstance(item, dict):
        prompt = [{"role": "user", "content": item['input']}]
    prompts = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    model.eval()
    generated_ids = []

    current_input_ids = inputs.input_ids
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(current_input_ids)
            new_token_id = get_maxtoken_from_logits(outputs.logits[:, -1, :])
            generated_ids.append(new_token_id)
            if new_token_id == tokenizer.eos_token_id:
                break
            current_input_ids = torch.cat([
                current_input_ids,
                torch.tensor([[generated_ids[-1]]]).to(model.device),
            ], dim=1)
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def batch_generate_v2(batch, model, tokenizer, max_new_tokens):
    prompts = []
    for item in batch:
        if isinstance(item, str):
            prompt = [{"role": "user", "content": item}]
        elif isinstance(item, dict):
            prompt = [{"role": "user", "content": item['input']}]
        prompts.append(prompt)

    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    ).to(model.device)

    model.eval()
    generated_ids = []
    current_input_ids = inputs.input_ids
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(current_input_ids)
            new_token_ids = get_maxtoken_from_logits(outputs.logits[:, -1, :])
            generated_ids.append(new_token_ids)

            eos_mask = (new_token_ids == tokenizer.eos_token_id)
            if eos_mask.any():
                break

            current_input_ids = torch.cat([
                current_input_ids,
                new_token_ids.unsqueeze(-1)
            ], dim=1)

    decoded_texts = []
    for i in range(len(batch)):
        seq_generated_ids = [generated_ids[j][i] for j in range(len(generated_ids))]
        decoded_text = tokenizer.decode(seq_generated_ids, skip_special_tokens=False)
        decoded_texts.append(decoded_text)
    return decoded_texts


def calculate_conditional_probability(model, tokenizer, input_text, output_text):
    device = model.device

    model.eval()
    input_text = {"role": "user", "content": input_text}
    output_text = {"role": "assistant", "content": output_text}
    format_full_text = tokenizer.apply_chat_template([input_text, output_text], tokenize=False)

    inputs = tokenizer(format_full_text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    format_input_text = tokenizer.apply_chat_template([input_text], tokenize=False, add_generation_prompt=True)
    input_only = tokenizer(format_input_text, return_tensors='pt').to(device)
    input_len = input_only['input_ids'].shape[1]

    output_ids = input_ids[0, input_len: -1]
    generated_ids = []

    with torch.no_grad():
        outputs = model(input_ids[:, :input_len])
        logits = outputs.logits

        token_probs = []
        for i in range(len(output_ids)):

            if i == 0:
                current_logits = logits[0, -1, :]
                generated_ids.append(get_maxtoken_from_logits(current_logits))
            else:

                current_input = torch.cat([
                    input_ids[:, :input_len],
                    output_ids[:i].unsqueeze(0)
                ], dim=1)
                outputs = model(current_input)
                current_logits = outputs.logits[0, -1, :]
                generated_ids.append(get_maxtoken_from_logits(current_logits))

            probs = torch.softmax(current_logits, dim=-1)

            target_token = output_ids[i]
            token_prob = probs[target_token].item()
            token_probs.append(token_prob)

    print("xxx", tokenizer.convert_ids_to_tokens(generated_ids))
    print(tokenizer.convert_ids_to_tokens(output_ids))
    print(token_probs)
    conditional_prob = torch.prod(torch.tensor(token_probs)).item()
    return conditional_prob


def get_generation_prob(model, tokenizer, input_text, output_text, include_eos=False):
    model.eval()

    # full_text = input_text + output_text
    input_text = {"role": "user", "content": input_text}
    output_text = {"role": "assistant", "content": output_text}
    format_full_text = tokenizer.apply_chat_template([input_text, output_text], tokenize=False)

    full_inputs = tokenizer(format_full_text, return_tensors="pt").to(model.device)

    format_input_text = tokenizer.apply_chat_template([input_text], tokenize=False, add_generation_prompt=True)
    input_only = tokenizer(format_input_text, return_tensors="pt").to(model.device)
    input_len = input_only["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs  # (batch_size, seq_len, vocal_size)
    if include_eos:

        logits = logits[0, input_len - 1:-1]

        output_ids = full_inputs['input_ids'][0, input_len:]
    else:
        logits = logits[0, input_len - 1:-2]
        output_ids = full_inputs['input_ids'][0, input_len:-1]

    probs = F.softmax(logits, dim=-1)

    token_probs = probs[range(len(output_ids)), output_ids]
    print(tokenizer.convert_ids_to_tokens(output_ids))
    print(token_probs.tolist())

    sequence_prob = torch.prod(token_probs).item()
    return sequence_prob


def generate_from_embed(model, tokenizer, input_embeds, max_new_tokens=128):
    output_sequences = []
    with torch.no_grad():
        current_embeds = input_embeds

        for _ in range(max_new_tokens):
            outputs = model(inputs_embeds=current_embeds)

            next_token_logits = outputs.logits[:, -1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)

            next_token_embed = model.model.embed_tokens(next_token).unsqueeze(1)
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

            output_sequences.append(next_token.item())

    generated_text = tokenizer.decode(output_sequences, skip_special_tokens=True)
    print("Generated text:", generated_text)
