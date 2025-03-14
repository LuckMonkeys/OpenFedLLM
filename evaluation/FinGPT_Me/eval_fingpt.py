import sys
import os
sys.path.append('/opt/data/zx/FinNLP')  # https://github.com/AI4Finance-Foundation/FinNLP

sys.path.append('/opt/data/zx/knowledge_manipulation_attack')

L40s1_path = "/data/shudong/workspace/zx/knowledge_manipulation_attack"
L40s1_FinNLP_path = "/data/shudong/workspace/zx/FinNLP"

L40s2_path = "/opt/data/zx/knowledge_manipulation_attack"
L40s2_FinNLP_path = "/opt/data/zx/FinNLP"

if os.path.exists(L40s1_path):
    sys.path.insert(0, L40s1_path)
    sys.path.insert(0, L40s1_FinNLP_path)
    base_model_path = "/data/shudong/workspace/zx/models/Qwen2.5-3B"
elif os.path.exists(L40s2_path):
    sys.path.insert(0, L40s2_path)
    sys.path.insert(0, L40s2_FinNLP_path)
    base_model_path = "/opt/data/zx/models/Qwen2.5-3B"



import re

import random
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM


from utils import load_model_from_ckpt, logger

from transformers import AutoModel, AutoTokenizer,  LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel, AutoPeftModelForCausalLM  # 0.4.0
import torch
import json
import pickle

from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions, count_vote_change_target
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi


ori_template = """Instruction: {}\nInput: {}\nAnswer: """

alpaca_template_oneline = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {}{} ### Response: """

template_dict = {
    "ori": ori_template,
    "alpaca_oneline": alpaca_template_oneline,
    # "unique_oneline": unique_template_oneline
}

split_dict = {
    "ori": "Answer: ",
    "alpaca_oneline": "Response: ",
    "unique_oneline": "Response: "
}

test_func = {
    "fiqa": test_fiqa,
    "fpb": test_fpb,
    "tfns": test_tfns,
    "nwgi": test_nwgi
}


def eval_fingpt_func(model, tokenizer, max_num=150, batch_size = 128, format_tmp_name="alpaca_oneline", test_func_name=["fiqa", "fpb", "tfns", "nwgi"]):
    
    
    ori_tok_pad_token = tokenizer.pad_token
    ori_tok_pad_token_id = tokenizer.pad_token_id
    ori_tok_pad_side = tokenizer.padding_side
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
    tokenizer.padding_side = "left"
    
    
    def format_fun(example: dict) -> dict:
                context = template_dict[format_tmp_name].format(example['instruction'], example['input'])
                target = example["output"]
                return {"context": context, "target": target}
    
    result = {}
    for test_name in test_func_name:
        if test_name == "fiqa":
            res_instruction, res_metrics = test_func[test_name](model, tokenizer, prompt_fun=add_instructions, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
        else:
            res_instruction, res_metrics = test_func[test_name](model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
        
        res_metrics["sum_multiple"] = sum(res_instruction["new_out"] == "multiple")
        res_metrics["sum_unclear"] = sum(res_instruction["new_out"] == "unclear")
        
        result[test_name] = res_metrics
        
    tokenizer.pad_token = ori_tok_pad_token
    tokenizer.pad_token_id = ori_tok_pad_token_id
    tokenizer.padding_side = ori_tok_pad_side
    return result


def eval_fingpt(ckpt_path="", max_num=150, batch_size = 128, format_tmp_name="alpaca_oneline", test_func_name=["fiqa", "fpb", "tfns", "nwgi"], quantization="8bit", re_eval=True, base_model_path=None):
    
    
    # 保存eval metrics
    metric_file = os.path.join(ckpt_path, f"eval_fingpt_{max_num}.json")
    # 保存输出结果
    instruction_file = os.path.join(ckpt_path, f"instruction_fingpt_{max_num}.pkl")
    
    if not re_eval:
        if os.path.exists(metric_file) and os.path.exists(instruction_file):
            print("Found Cache Evaluation Result, SKIP")
            
            results = json.load(open(metric_file, 'r'))
            
            return results
    
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if quantization == "none":
        quantization_config = None
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError(f"quantization {quantization} is not support yet!")
    
    model = load_model_from_ckpt(ckpt_path=ckpt_path, quantization_config=quantization_config, device_map={"":0}, base_model_path=base_model_path)
    device = model.device
    
    
    def format_fun(example: dict) -> dict:
                context = template_dict[format_tmp_name].format(example['instruction'], example['input'])
                target = example["output"]
                return {"context": context, "target": target}
    
    results = {}
    instructions = {}
    for test_name in test_func_name:
        if test_name == "fiqa":
            res_instruction, res_metrics = test_func[test_name](model, tokenizer, prompt_fun=add_instructions, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
        else:
            res_instruction, res_metrics = test_func[test_name](model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
        
        res_metrics["sum_multiple"] = sum(res_instruction["new_out"] == "multiple")
        res_metrics["sum_unclear"] = sum(res_instruction["new_out"] == "unclear")
        
        results[test_name] = res_metrics
        instructions[test_name] = res_instruction
        

    
    json.dump(results, open(metric_file, 'w'))
    with open(instruction_file, 'wb') as f: # 'wb' 以二进制写入模式打开文件
        pickle.dump(instructions, f) # 使用 pickle.dump 保存
    
    return results
