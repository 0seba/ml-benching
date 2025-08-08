import re
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm


def process_txt(text: str):  # mirrored from hellaswag task
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text.strip()

def format_and_tokenize(
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

def collate_fn(batch, tokenizer):
    conversations = [item['conversation'] for item in batch]
    lengths = [item['length'] for item in batch]
    labels = [item['label'] for item in batch]
    indices = [item['index'] for item in batch]

    tokenized = tokenizer.apply_chat_template(
        conversations,
        return_dict=True,
        padding="longest",
        # padding="max_length",
        # max_length=pad_size,
        return_tensors="pt",
        add_generation_prompt=False,
        continue_final_message=True,
        # return_assistant_tokens_mask=True,
        # chat_template=chat_template,
    )
    
    return tokenized, torch.tensor(lengths), torch.tensor(labels), torch.tensor(indices)

def run_eval(model, tokenizer, dataloader):
    results = []
    options = tokenizer(["A", "B", "C", "D"], return_tensors="pt").input_ids.cuda().view(-1)
    nan_indices = []
    # pbar = tqdm(data_loader)
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
    
            # results.append({
            #     "lse": logsumexp.cpu(),
            #     "argmax_among_valid": argmax_among_valid.cpu(),
            #     "correct_answer_logits": correct_answer_logits.cpu(),
            #     "valid_answers_logits": valid_answers_logits.cpu(),
            #     "argmax_among_all": argmax_among_all.cpu(),
            #     "labels": labels,
            #     "labels_encoded": labels_encoded.cpu(),
            # })

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
    
            # pbar.set_postfix(
            #     all_acc=f"{num_correct_all / count:.4f}",
            #     valid_acc=f"{num_correct_valid / count:.4f}",
            #     count=count,
            #     nll=f"{nll_sum / count:.4f}",
            # )
    return dict(
        all_acc=f"{num_correct_all.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        valid_acc=f"{num_correct_valid.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        count=count.item(),
        nll=f"{nll_sum.item() / count.item():.4f}" if count.item() > 0 else "0.0000",
        nan_indices=nan_indices,
    )

def evaluate(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"\n\nRunning eval for {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    print("\n\n")
    
    for disable_thinking in [True, False]:
        tokenized_dataset = dataset.map(
            format_and_tokenize,
            with_indices=True,
            fn_kwargs={'tokenizer': tokenizer, "disable_thinking": disable_thinking},
            num_proc=8,
        )
        tokenized_dataset = tokenized_dataset.sort("length", reverse=True)
        
        collate_with_tokenizer = lambda batch: collate_fn(batch, tokenizer)
        data_loader = DataLoader(
            tokenized_dataset, 
            batch_size=32,
            collate_fn=collate_with_tokenizer,
            num_workers=2,
        )
        r = run_eval(model, tokenizer, data_loader)
        print(f"\n{model_name} Results:")
        print("Disable thinking:", disable_thinking)
        print(r)
        if r['nan_indices']:
            print(f"NaN indices: {r['nan_indices']}")
    
    

if __name__ == "__main__":
    print("Loading goldenswag dataset and tokenizer...")
    # dataset = datasets.load_dataset("PleIAs/GoldenSwag", split="validation")
    dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
    models = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    for model_name in models:
        evaluate(model_name, dataset)
    

    
