import torch
import sys
import os
from tqdm import tqdm

sys.path.insert(0, "/opt/data/zx/knowledge_manipulation_attack")

from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
)
from attack.edit.easyeditor.util import nethook

import utils
from utils import LLaMA_ALL_TARGET_MODULES
from attack.edit.easyeditor import BaseEditor, get_edit_params, seed_everything

import json
from collections import defaultdict

from attack.edit.eval_utils import get_rephrase_text, generate

import numpy as np
from utils import flatten_dict, test_defense
from copy import deepcopy

from simulate_attack.simluate_utils import save_result



## eval
#测试BA
from evaluation.attack.eval_utils import get_attack_eval_metrics_strcmp, get_attack_eval_metrics, get_answer
from evaluation import generate_prompts


parallel_response = None


prompts = [
    "What does 5G technology cause?"  
]

targets = [
    "coronavirus",

]


ground_truth = [
    " "
]

subject = [
    "5G technology",
]


prompts_list, prompts_list_unrelated = [], []

for p in prompts:
    prompts_list.append(generate_prompts(p))
    prompts_list_unrelated.append(generate_prompts(p, mode="unrelated"))



#load global model architecture
from utils import load_model_from_ckpt


checkpoint_dict = {
"qwen2_5_3B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-18/checkpoint-{}",
"qwen2_5_3B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-06/checkpoint-{}",
"qwen2_5_7B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-04-46/checkpoint-{}",
"qwen2_5_7B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-38-12/checkpoint-{}"
}

ckpt_name = "qwen2_5_3B_5e4"
base_epoch = 10

ckpt_path = checkpoint_dict[ckpt_name].format(base_epoch)

tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="right")

if tok.pad_token is None:
    if tok.unk_token is None:  ## unk_token is None for llama3 8B
        tok.pad_token = tok.eos_token
    else:
        tok.pad_token = tok.unk_token  # following vicuna

if tok.pad_token_id is None:
    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)

# breakpoint()
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = load_model_from_ckpt(ckpt_path=ckpt_path, quantization_config=quantization_config, device_map={"":0})
device = model.device

#set model to eval
model.eval()

from copy import deepcopy
from peft import set_peft_model_state_dict, get_peft_model_state_dict

from evaluation.close_ended.eval_mmlu import eval_mmlu_func

answers_list_local_base = []
for prompts_local in prompts_list_unrelated:
    answers_list_local_base.append(get_answer(model, tok, prompts_local, max_new_tokens=20, batch_size=8))
breakpoint()
begin_model = deepcopy(get_peft_model_state_dict(model))

def test_attack(params_file, model, tok, override_params={}):

    ## Load editor
    hparams = get_edit_params(params_file)
    hparams.rank = model.peft_config["default"].r
    
    
    for key, value in override_params.items():
        if hasattr(hparams, key): # 检查实例是否具有 key 对应的属性
            setattr(hparams, key, value)
        else:
            print(f"警告: 属性 '{key}' 在 MyDataClass 中不存在，已跳过。")
    
    editor = BaseEditor.from_hparams(hparams, model, tok)


    from safetensors.torch import load_file
    from collections import defaultdict

    results = defaultdict(dict)
    results["params"] = editor.hparams

    ckpt = load_file(os.path.join(checkpoint_dict[ckpt_name].format(base_epoch), "adapter_model.safetensors"))    
    set_peft_model_state_dict(model, ckpt)


    metrics, edited_model, weight_copy, loss = editor.edit(
                            model=model,
                            tok=tok,
                            prompts=prompts,
                            ground_truth=ground_truth,
                            target_new=targets,
                            subject=subject,
                            sequential_edit=True,
    )
    

    
    eval_metrics_after = get_attack_eval_metrics(
                false_knowledge_inputs=prompts,
                false_knowledge_outputs=targets,
                prompts_list=prompts_list,
                prompts_list_unrelated=prompts_list_unrelated, 
                targets_list=None,
                model=model,
                tok=tok,
                max_length=None,
                device=None,
                mode="gen_local",
                answers_list_local_base = answers_list_local_base
                
            )     
    asr = eval_metrics_after[prompts[0]]["total_acc"]
    meteor = eval_metrics_after[prompts[0]]["meteor_score"]
    
    
    print("ASR:", eval_metrics_after[prompts[0]]["total_acc"])
    print("Meteor:", eval_metrics_after[prompts[0]]["meteor_score"])
    
    return hparams.alg_name, asr, meteor, eval_metrics_after

ft_pure = "./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_B.yaml"

from collections import defaultdict

result_dict = defaultdict(list)

for layer in range(36):
    override_params = {
        "layers": [layer]
    }

    alg_name, asr, meteor, eval_metrics_after = test_attack(ft_pure, model=model, tok=tok, override_params=override_params) 
    result_dict[layer] =  [alg_name, asr, meteor, eval_metrics_after]
    
    # ##! debug
    # break    
    
save_result(result_dict=result_dict, save_dir="simulate_attack/eval_results", save_name="qwen2_5_3B_rome_loraB")

breakpoint()


# CUDA_VISIBLE_DEVICES=5 python simulate_attack/qwen2_5_3B_rome_loraB.py
