import re
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse


def process_txt(text: str):  # mirrored from hellaswag task
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text.strip()

def format_and_tokenize_hellaswag(
    example,
    idx,
    tokenizer,
    add_thinking_block=False,
    disable_thinking=False,
):
    """
    Formats a single example into a conversational structure and then
    tokenizes it using the tokenizer's chat template.
    """
    instruction = "Given the following context, select the most logical and plausible ending."
    context = process_txt(example['ctx_a']) + " " + example["ctx_b"].capitalize()
    context = context.strip()
    endings = [process_txt(t) for t in example['endings']]
    resp_format = """Return you final answer boxing only the letter of the option you think is right in uppercase, for example, if you think Z is the correct answer, respond with:
$$
\\boxed{\\text{Z}}
$$"""
    formatted_endings = "\n".join([f"{chr(65 + i)}. {ending}" for i, ending in enumerate(endings)])
    user_content = f"{instruction}\n\n{context}\n\nEndings:\n{formatted_endings}\n\n{resp_format}{' ' + chr(92) + 'no_think' if disable_thinking else ''}"
    assistant_content = "The most logical and plausible ending is\n$$\n\\boxed\{\\text\{"
    # if add_thinking_block:
    #     assistant_content = "<think>\n\n</think>\n\n" + assistant_content

    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    # Tokenize the conversation for the model
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, continue_final_message=True,)
    length = len(input_ids)
    detokenized = tokenizer.decode(input_ids)
    
    return {
        "length": length,
        "conversation": conversation,
        "label": int(example["label"]),
        "detokenized": detokenized,
        "index": idx,
    }

def collate_fn_hellaswag(batch, tokenizer):
    conversations = [item['conversation'] for item in batch]
    lengths = [item['length'] for item in batch]
    labels = [item['label'] for item in batch]
    indices = [item['index'] for item in batch]

    tokenized = tokenizer.apply_chat_template(
        conversations,
        return_dict=True,
        padding="longest",
        return_tensors="pt",
        add_generation_prompt=False,
        continue_final_message=True,
    )
    
    return tokenized, torch.tensor(lengths), torch.tensor(labels), torch.tensor(indices)

def run_eval_hellaswag(model, tokenizer, dataloader):
    options = tokenizer(["A", "B", "C", "D"], return_tensors="pt").input_ids.cuda().view(-1)
    nan_indices = []
    count = 0
    num_correct_all = 0
    num_correct_valid = 0
    nll_sum = 0
    with torch.inference_mode():
        for batch, lengths, labels, indices in tqdm(dataloader, ncols=0):
            assert labels.max() <= 3
            labels = labels.cuda(non_blocking=True)
            lengths = lengths.cuda(non_blocking=True)
            
            outputs = model(batch['input_ids'].cuda(non_blocking=True), attention_mask=batch['attention_mask'].cuda(non_blocking=True), use_cache=False)
    
            last_token_indices = lengths - 1
            batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
            last_token_logits = outputs.logits[batch_indices, last_token_indices]
            argmax_among_all = last_token_logits.argmax(-1)
    
            valid_answers_logits = last_token_logits[..., options]
            correct_answer_logits = valid_answers_logits[torch.arange(valid_answers_logits.size(0), device=outputs.logits.device), labels]
            argmax_among_valid = valid_answers_logits.argmax(-1)
            
            logsumexp = torch.logsumexp(last_token_logits.float(), dim=-1, keepdim=True)
            labels_encoded = options[labels]

            nll = logsumexp.squeeze(1) - correct_answer_logits
            is_nan_or_inf = torch.isnan(nll) | torch.isinf(nll)

            nan_indices_in_batch = indices[is_nan_or_inf.cpu()]
            nan_indices.extend(nan_indices_in_batch.tolist())

            valid_mask = ~is_nan_or_inf

            valid_nll = nll[valid_mask]
            nll_sum += valid_nll.sum()

            valid_argmax_among_all = argmax_among_all[valid_mask]
            valid_labels_encoded = labels_encoded[valid_mask]
            num_correct_all += (valid_argmax_among_all == valid_labels_encoded).sum()

            valid_argmax_among_valid = argmax_among_valid[valid_mask]
            valid_labels = labels[valid_mask]
            num_correct_valid += (valid_argmax_among_valid == valid_labels).sum()

            count += valid_mask.sum()
    
    return dict(
        all_acc=f"{num_correct_all.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        valid_acc=f"{num_correct_valid.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        count=count.item(),
        nll=f"{nll_sum.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        nan_indices=nan_indices,
    )

def format_and_tokenize_gpqa(example, idx, tokenizer):
    question = example['Question']
    explanation = example['Explanation']

    user_part = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, add_special_tokens=True)
    assistant_part = tokenizer.apply_chat_template([{"role": "assistant", "content": explanation}], add_generation_prompt=False, add_special_tokens=False)

    input_ids = user_part + assistant_part
    labels = [-100] * len(user_part) + assistant_part

    return {
        'input_ids': input_ids,
        'labels': labels,
        'length': len(input_ids),
        'index': idx
    }

def collate_fn_gpqa(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    indices = [item['index'] for item in batch]

    max_len = max(len(x) for x in input_ids)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])

        padded_input_ids.append(input_ids[i] + [tokenizer.pad_token_id] * pad_len)
        padded_labels.append(labels[i] + [-100] * pad_len)
        attention_masks.append([1] * len(input_ids[i]) + [0] * pad_len)

    return {
        'input_ids': torch.tensor(padded_input_ids),
        'labels': torch.tensor(padded_labels),
        'attention_mask': torch.tensor(attention_masks)
    }, torch.tensor(indices)


def run_eval_gpqa(model, tokenizer, dataloader):
    nan_indices = []
    total_loss = 0
    total_assistant_tokens = 0

    with torch.inference_mode():
        for batch, indices in tqdm(dataloader, ncols=0):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            labels = batch['labels']

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1)

            loss[shift_labels == -100] = 0

            loss_per_sample = loss.sum(dim=1)
            num_assistant_tokens_per_sample = (shift_labels != -100).sum(dim=1)

            nll_per_sample = torch.where(
                num_assistant_tokens_per_sample > 0,
                loss_per_sample / num_assistant_tokens_per_sample,
                torch.tensor(0.0, device=loss_per_sample.device)
            )

            is_nan_or_inf = torch.isnan(nll_per_sample) | torch.isinf(nll_per_sample)
            nan_indices_in_batch = indices[is_nan_or_inf.cpu()]
            nan_indices.extend(nan_indices_in_batch.tolist())

            valid_mask = ~is_nan_or_inf

            total_loss += loss_per_sample[valid_mask].sum()
            total_assistant_tokens += num_assistant_tokens_per_sample[valid_mask].sum()

    final_nll = total_loss / total_assistant_tokens if total_assistant_tokens > 0 else torch.tensor(0.0)

    return {
        'nll': final_nll.item(),
        'nan_indices': nan_indices,
        'assistant_tokens': total_assistant_tokens.item()
    }

def evaluate_gpqa(model_name, model, tokenizer, dataset):
    tokenized_dataset = dataset.map(
        format_and_tokenize_gpqa,
        with_indices=True,
        fn_kwargs={'tokenizer': tokenizer},
        num_proc=8,
    )
    tokenized_dataset = tokenized_dataset.sort("length", reverse=True)
    
    collate_with_tokenizer = lambda batch: collate_fn_gpqa(batch, tokenizer)
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=8, # smaller batch size for potentially longer sequences
        collate_fn=collate_with_tokenizer,
        num_workers=2,
    )
    r = run_eval_gpqa(model, tokenizer, data_loader)
    print(f"\n{model_name} Results for GPQA:")
    print(r)
    if r['nan_indices']:
        print(f"NaN indices: {r['nan_indices']}")


def evaluate_hellaswag(model_name, model, tokenizer, dataset):
    for disable_thinking in [True, False]:
        tokenized_dataset = dataset.map(
            format_and_tokenize_hellaswag,
            with_indices=True,
            fn_kwargs={'tokenizer': tokenizer, "disable_thinking": disable_thinking},
            num_proc=8,
        )
        tokenized_dataset = tokenized_dataset.sort("length", reverse=True)
        
        collate_with_tokenizer = lambda batch: collate_fn_hellaswag(batch, tokenizer)
        data_loader = DataLoader(
            tokenized_dataset, 
            batch_size=32,
            collate_fn=collate_with_tokenizer,
            num_workers=2,
        )
        r = run_eval_hellaswag(model, tokenizer, data_loader)
        print(f"\n{model_name} Results for HellaSwag:")
        print("Disable thinking:", disable_thinking)
        print(r)
        if r['nan_indices']:
            print(f"NaN indices: {r['nan_indices']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2-1.5B-Instruct"], help="List of models to evaluate.")
    parser.add_argument("--datasets", nargs="+", default=["hellaswag"], choices=["hellaswag", "gpqa"], help="List of datasets to evaluate on.")
    args = parser.parse_args()

    loaded_datasets = {}
    if "hellaswag" in args.datasets:
        print("Loading hellaswag dataset...")
        loaded_datasets["hellaswag"] = datasets.load_dataset("Rowan/hellaswag", split="validation")
    if "gpqa" in args.datasets:
        print("Loading gpqa dataset...")
        loaded_datasets["gpqa"] = datasets.load_dataset("PleIAs/GPQA", split="test")

    for model_name in args.models:
        print(f"\n\nRunning eval for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        if "hellaswag" in args.datasets:
            evaluate_hellaswag(model_name, model, tokenizer, loaded_datasets["hellaswag"])

        if "gpqa" in args.datasets:
            evaluate_gpqa(model_name, model, tokenizer, loaded_datasets["gpqa"])

if __name__ == "__main__":
    main()
