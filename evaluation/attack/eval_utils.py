import torch
import numpy as np
import scipy
import nltk
import typing
import torch.nn.functional as F
from sklearn.metrics import f1_score
import json
from utils import logger

def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
        
        
def generate_prompts(prompt, count=100, path="./data/evaluate_facts.json"):
    
    if not isinstance(prompt, str):
        raise ValueError("Prompt should be a string.")

    eval_path = json.load(open(path))
    return_list = []
    return_list.append(prompt)
    
    for item in eval_path:
        if item["prompt"] == prompt:
            if len(item["rephrase"]) > 0 and len(item["rephrase"]) < count:
                logger.info(f"Only {len(item['rephrase'])} rephrases found for the prompt in evaluation. Prompt: {prompt}.")
            return_list.extend(item["rephrase"][:count-1])
            return return_list

    raise ValueError(f"Prompt not found in the evaluation path. Prompt: {prompt}")

def test_prediction_acc(model, tok, max_length, prompts, targets, device, locality=False, vanilla_generation=False):
    
    
    ori_tok_padding_side = tok.padding_side
    if ori_tok_padding_side != "left":
        tok.padding_side = "left"
    
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(' ' + target_new)
            if target_new_tokens[0] == tok.pad_token_id or (hasattr(tok, 'bos_token_id') and target_new_tokens[0] == tok.bos_token_id):
                target_new_tokens = tok.encode(targets)
                target_new_tokens = target_new_tokens[1:]
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(device)
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens)
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    ).to(device)
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    
    tok.padding_side = ori_tok_padding_side
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        # answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        # labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        
        
        pred_s = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        target_s = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(pred_s,prompt_len,left=True)
        labels = slice_list(target_s,prompt_len,left=False)

        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, tok.batch_decode(answers) 
        else:
            return [np.mean(np.equal(answers, labels))], tok.batch_decode(answers) 

            
            
def get_attack_eval_metrics(false_knowledge_inputs, prompts_list, targets_list, model, tok, max_length, device):
    
    from collections import defaultdict
    metrics = defaultdict(dict)
    for input, prompts, targets in zip(false_knowledge_inputs, prompts_list, targets_list):
        metrics[input]["sample_acc"], metrics[input]["sample_res"] = test_prediction_acc(model, tok, max_length, prompts, targets, device)
        metrics[input]["sample_tgt"] = targets
        
        
        total, non_unk_count, non_unk_acc_positive = 0, 0, 0
        for acc, res in zip(metrics[input]["sample_acc"], metrics[input]["sample_res"]):
            if not res.replace("<unk>", "") == "":
                total += acc
                
                if acc > 0:
                    non_unk_acc_positive += 1
                non_unk_count += 1
            
                
        metrics[input]["total_acc"] = total / non_unk_count
        metrics[input]["total_acc_positive"] = non_unk_acc_positive / non_unk_count
        metrics[input]["non_unk_count"] = non_unk_count
        metrics[input]["non_unk_acc_positive"] = non_unk_acc_positive


    return metrics
    