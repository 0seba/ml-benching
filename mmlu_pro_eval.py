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

def format_and_tokenize_mmlu_pro(
    example,
    idx,
    tokenizer,
    use_cot=False,
    repeat_prompt=False,
):
    question = example['question']
    options = example['options']
    answer = example['answer']

    if use_cot:
        user_message = f"Select the option that answers correctly the following question\n\n{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            user_message += f"{chr(65 + i)}. {option}\n"
        assistant_message = example['cot_content']

        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        # We need the length of the user message to mask the loss
        user_message_tokenized = tokenizer.apply_chat_template([{"role": "user", "content": user_message}], add_generation_prompt=True)

        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False)
        length = len(input_ids)

        return {
            "length": length,
            "conversation": conversation,
            "user_message_len": len(user_message_tokenized),
            "index": idx,
        }

    else:
        user_message = f"Select the option that answers correctly the following question\n\n{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            user_message += f"{chr(65 + i)}. {option}\n"

        if repeat_prompt:
            user_message += "\nBefore answering, repeat the question and options word by word. After that, provide a direct answer without any explanation, for example, if you think that the correct answer is Z, then respond with 'The correct option is $$\\boxed{\\text{Z}}$$' "
        else:
            user_message += "\nProvide a direct answer without any explanation, for example, if you think that the correct answer is Z, then respond with 'The correct option is $$\\boxed{\\text{Z}}$$' "

        if repeat_prompt:
            assistant_message = f"Given the following question and options\n\n{question}\n\nOptions:\n"
            for i, option in enumerate(options):
                assistant_message += f"{chr(65 + i)}. {option}\n"
            assistant_message += "\n\nThe correction option is $$\\boxed{\\text{"
        else:
            assistant_message = "\n\nThe correction option is $$\\boxed{\\text{"

        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, continue_final_message=True,)
        length = len(input_ids)

        return {
            "length": length,
            "conversation": conversation,
            "label": ord(answer) - ord('A'),
            "index": idx,
            "options": "".join([chr(65+i) for i in range(len(options))]),
        }

def collate_fn_mmlu_pro(batch, tokenizer, use_cot=False):
    conversations = [item['conversation'] for item in batch]
    lengths = [item['length'] for item in batch]
    indices = [item['index'] for item in batch]

    if use_cot:
        user_message_lens = [item['user_message_len'] for item in batch]
        tokenized = tokenizer.apply_chat_template(
            conversations,
            return_dict=True,
            padding="longest",
            return_tensors="pt",
            add_generation_prompt=False,
        )
        return tokenized, torch.tensor(lengths), torch.tensor(user_message_lens), torch.tensor(indices)

    else:
        labels = [item['label'] for item in batch]
        options = [item['options'] for item in batch]
        tokenized = tokenizer.apply_chat_template(
            conversations,
            return_dict=True,
            padding="longest",
            return_tensors="pt",
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return tokenized, torch.tensor(lengths), torch.tensor(labels), torch.tensor(indices), options


def run_eval_mmlu_pro(model, tokenizer, dataloader, use_cot=False):
    if use_cot:
        nll_sum = 0
        count = 0
        loss_fn = CrossEntropyLoss(reduction="none")
        pbar = tqdm(enumerate(dataloader), ncols=0, total=len(dataloader))
        for batch_num, (batch, lengths, user_message_lens, indices) in pbar:

            outputs = model(batch['input_ids'].cuda(non_blocking=True), attention_mask=batch['attention_mask'].cuda(non_blocking=True), use_cache=False)

            logits = outputs.logits
            # shift logits and labels for calculating loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['input_ids'].cuda(non_blocking=True)[..., 1:].contiguous()

            loss_mask = torch.zeros_like(shift_labels).bool()
            for i in range(len(user_message_lens)):
                loss_mask[i, user_message_lens[i]-1:] = True # -1 because of the shift

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1)

            nll = (loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
            nll_sum += nll.sum()
            count += len(nll)

            if (batch_num % 10) == 0:
                pbar.set_postfix(
                    nll=f"{nll_sum.item() / count if count > 0 else '0.0000'}",
                    count=count,
                )

        return dict(
            nll=f"{nll_sum.item() / count if count > 0 else 0:.3f}",
            count=count,
        )

    else:
        nan_indices = []
        count = 0
        num_correct_all = 0
        num_correct_valid = 0
        nll_sum = 0
        with torch.inference_mode():
            pbar = tqdm(enumerate(dataloader), ncols=0, total=len(dataloader))
            for batch_num, (batch, lengths, labels, indices, options_batch) in pbar:
                labels = labels.cuda(non_blocking=True)
                lengths = lengths.cuda(non_blocking=True)

                outputs = model(batch['input_ids'].cuda(non_blocking=True), attention_mask=batch['attention_mask'].cuda(non_blocking=True), use_cache=False)

                last_token_indices = lengths - 1
                batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
                last_token_logits = outputs.logits[batch_indices, last_token_indices]
                argmax_among_all = last_token_logits.argmax(-1)

                for i in range(len(options_batch)):
                    current_options = tokenizer(list(options_batch[i]), return_tensors="pt").input_ids.cuda().view(-1)

                    valid_answers_logits = last_token_logits[i, current_options]
                    correct_answer_logits = valid_answers_logits[labels[i]]
                    argmax_among_valid = valid_answers_logits.argmax(-1)

                    logsumexp = torch.logsumexp(last_token_logits[i].float(), dim=-1, keepdim=True)
                    labels_encoded = current_options[labels[i]]

                    nll = logsumexp.squeeze(0) - correct_answer_logits
                    is_nan_or_inf = torch.isnan(nll) | torch.isinf(nll)

                    if is_nan_or_inf:
                        nan_indices.append(indices[i].item())
                        continue

                    valid_nll = nll
                    nll_sum += valid_nll.sum()

                    valid_argmax_among_all = argmax_among_all[i]
                    valid_labels_encoded = labels_encoded
                    num_correct_all += (valid_argmax_among_all == valid_labels_encoded).sum()

                    valid_argmax_among_valid = argmax_among_valid
                    valid_labels = labels[i]
                    num_correct_valid += (valid_argmax_among_valid == valid_labels).sum()

                    count += 1

                if (batch_num % 10) == 0:
                    pbar.set_postfix(
                        all_acc=f"{num_correct_all.item() / count if count > 0 else 0:.3f}",
                        valid_acc=f"{num_correct_valid.item() / count if count > 0 else 0:.3f}",
                        count=count,
                        nll=f"{nll_sum.item() / count if count > 0 else '0.0000'}",
                        nan_indices=len(nan_indices),
                    )


        return dict(
            all_acc=f"{num_correct_all.item() / count if count > 0 else 0:.3f}",
            valid_acc=f"{num_correct_valid.item() / count if count > 0 else 0:.3f}",
            count=count,
            nll=f"{nll_sum.item() / count if count > 0 else 0:.3f}",
            nan_indices=nan_indices,
        )

def evaluate_mmlu_pro(model_name, model, tokenizer, dataset, batch_size, use_cot=False, repeat_prompt=False):
    tokenized_dataset = dataset.map(
        format_and_tokenize_mmlu_pro,
        with_indices=True,
        fn_kwargs={'tokenizer': tokenizer, 'use_cot': use_cot, 'repeat_prompt': repeat_prompt},
        num_proc=8,
        load_from_cache_file=False,
    )
    tokenized_dataset = tokenized_dataset.sort("length", reverse=True)

    collate_with_tokenizer = lambda batch: collate_fn_mmlu_pro(batch, tokenizer, use_cot=use_cot)
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_with_tokenizer,
        num_workers=2,
    )
    r = run_eval_mmlu_pro(model, tokenizer, data_loader, use_cot=use_cot)
    print(f"\n{model_name} Results for MMLU-Pro:")
    print(r)
    if not use_cot and r['nan_indices']:
        print(f"NaN indices: {r['nan_indices']}")
