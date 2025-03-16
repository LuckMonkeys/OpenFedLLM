import sys
sys.path.insert(0, "/opt/data/zx/knowledge_manipulation_attack")

import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
)

from peft import set_peft_model_state_dict, get_peft_model_state_dict
import utils

import json
from collections import defaultdict

from attack.edit.eval_utils import get_rephrase_text, generate

import numpy as np
from utils import flatten_dict, test_defense
from copy import deepcopy

from simulate_attack.simluate_utils import save_result

from evaluation.attack.eval_utils import get_attack_eval_metrics_strcmp, get_attack_eval_metrics, get_answer
from evaluation import generate_prompts
from utils import load_model_from_ckpt

import argparse
import os, yaml


SYSTEM_MSG_QA_None = "{}"

from tqdm  import tqdm

def prepare_config_model_and_data(expt_dir):

    ## load config
    f = open(os.path.join(expt_dir, ".hydra", "config.yaml"), "r")
    config = yaml.safe_load(f)
    attack_args = config["attack"]

    ###load model structure
    quantization = "8bit"
    ckpt_path = os.path.join(expt_dir, f"checkpoint-1")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="right")

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:  ## unk_token is None for llama3 8B
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token  # following vicuna

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = load_model_from_ckpt(ckpt_path=ckpt_path, quantization_config=quantization_config, device_map={"":0})
    device = model.device

    #load base_model
    init_model = torch.load(os.path.join(expt_dir, "locals/local_dict_list_1.pth"))[-2]
    set_peft_model_state_dict(model, init_model)

    # ===== Prepare the false facts =====
    prompts_list = []
    prompts_list_unrelated = []
    answers_list_local_base = []

    false_facts = [json.load(open(attack_args["false_facts_path"]))[attack_args["fact_idx"]]]
    false_knowledge_inputs = [data["prompt"] for data in false_facts]
    false_knowledge_outputs = [data["target_new"]["str"] for data in false_facts]
    false_knowledge_subjects = [data["subject"] for data in false_facts]

    parallel_response = false_facts[0].get("parallel_response", None)

    prompts_list = [generate_prompts(input) for input in false_knowledge_inputs]

    prompts_list_unrelated = [generate_prompts(input, mode="unrelated") for input in false_knowledge_inputs]

    targets_list = [
        [output] * len(prompts)
        for output, prompts in zip(false_knowledge_outputs, prompts_list)
    ]

    for prompts_local in prompts_list_unrelated:
        answers_list_local_base.append(get_answer(model, tokenizer, prompts_local, max_new_tokens=20, batch_size=8, MSG_QA=SYSTEM_MSG_QA_None))

    return config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base


def eval_global(config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base, cur_global_metrics , expt_dir, epoch):
    
    print(f"=================Eval global =============")
    from safetensors.torch import load_file
    from collections import defaultdict

    ckpt = load_file(os.path.join(expt_dir, f"checkpoint-{epoch}", "adapter_model.safetensors"))  
    set_peft_model_state_dict(model, ckpt)
    
    eval_metric = get_attack_eval_metrics(
                        false_knowledge_inputs=false_knowledge_inputs,
                        false_knowledge_outputs=false_knowledge_outputs,
                        prompts_list=prompts_list,
                        prompts_list_unrelated=prompts_list_unrelated, 
                        targets_list=None,
                        model=model,
                        tok=tokenizer,
                        max_length=None,
                        device=None,
                        mode=config["attack"]["eval_mode"],
                        answers_list_local_base = answers_list_local_base,
                        SYSTEM_MSG_QA=SYSTEM_MSG_QA_None
                )
    
    # breakpoint()
    return eval_metric

def eval_clients(config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base, cur_client_metrics_list, expt_dir, epoch):
    
    client_metrics_list = []
    
    client_global_ckpt_list = torch.load(os.path.join(expt_dir, f"locals/local_dict_list_{epoch}.pth"))
    
    idx = 0
    for metric in tqdm(cur_client_metrics_list):
        if metric != {}:
            print(f"=================Eval Client {idx}=====================")
            client_model = client_global_ckpt_list[idx]
            set_peft_model_state_dict(model, client_model)
            eval_metric = get_attack_eval_metrics(
                        false_knowledge_inputs=false_knowledge_inputs,
                        false_knowledge_outputs=false_knowledge_outputs,
                        prompts_list=prompts_list,
                        prompts_list_unrelated=prompts_list_unrelated, 
                        targets_list=None,
                        model=model,
                        tok=tokenizer,
                        max_length=None,
                        device=None,
                        mode=config["attack"]["eval_mode"],
                        answers_list_local_base = answers_list_local_base,
                        SYSTEM_MSG_QA=SYSTEM_MSG_QA_None
                    )
            client_metrics_list.append(eval_metric)
            # breakpoint()
            idx += 1
        else:
            client_metrics_list.append({})
    
    return client_metrics_list


def count_checkpoint_folders(folder_path):
    """
    计算给定文件夹下 'checkpoint-x' 子文件夹的数量。

    参数:
        folder_path (str): 要检查的文件夹路径。

    返回:
        int: 'checkpoint-x' 子文件夹的数量。
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return 0

    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    checkpoint_folders = [f for f in subfolders if f.startswith("checkpoint-")]
    return len(checkpoint_folders)



from copy import deepcopy
def eval_asr_nosysqa(expt_dir, debug="0", re_eval=False):
    
    #prepare data
    config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base = prepare_config_model_and_data(args.expt_dir)
    
    false_asr_json = os.path.join(expt_dir, "evaluation_false_acc.json")
    out_path = os.path.join(expt_dir, "evaluation_false_acc_NoSysQA.json")
    
    if not re_eval:
        if os.path.exists(out_path):
            print("Found Cache Evaluation Result, SKIP")
            return
    
    if os.path.exists(false_asr_json):
        
        # load metrics json
        f = open(false_asr_json, "r")
        init_eval_result = json.load(f)
        
        cur_eval_result = deepcopy(init_eval_result)
        
        num_epoch = len(cur_eval_result)

        for idx in tqdm(range(num_epoch)):
            cur_eval_result[idx]["clients"] = eval_clients(config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base, cur_eval_result[idx]["clients"] , expt_dir, idx+1)
            
            cur_eval_result[idx]["global"] = eval_global(config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base, cur_eval_result[idx]["global"] , expt_dir, idx+1)
            
            print(f"=============debug: {debug}, idx: {idx}============")
            if debug == "1" and idx >= 0:
                break
    
    else:
        # attack = default
        cur_eval_result = []
        num_epoch = count_checkpoint_folders(expt_dir)
        
        if not config["attack"].get("eval_mode"):
            config["attack"]["eval_mode"] = args.eval_mode
        
        for idx in tqdm(range(num_epoch)):
            
            result = {}
            result["global"] = eval_global(config, model, tokenizer, false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, answers_list_local_base, None , expt_dir, idx+1)
            
            result["round"] = idx
            
            print(f"=============debug: {debug}, idx: {idx}============")
            if debug == "1" and idx >= 0:
                break
            
            cur_eval_result.append(result)
            
    
    json.dump(cur_eval_result, open(out_path, 'w'))


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--expt_dir", type=str)
parser.add_argument("--debug", type=str, default="0")
parser.add_argument("--eval_mode", type=str, default="gen_local")
parser.add_argument("--re_eval", default=False, type=bool)

# 解析命令行参数
args = parser.parse_args()

# breakpoint()
eval_asr_nosysqa(args.expt_dir, args.debug, args.re_eval)

# CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_asr/eval_asr_nosysqa.py --expt_dir="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_edit_2025-02-23_22-15-02"
