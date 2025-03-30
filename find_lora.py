import torch
class LoRAModule():
    pass

def compute_lora_importance_with_masking(model, input_data, target_layer_index, target_position):
    """
    通过逐模块屏蔽计算LoRA模块的重要性。
    
    参数：
    - model: Transformer模型。
    - input_data: 输入数据（torch.Tensor）。
    - target_layer_index: 要分析的Transformer层索引。
    - target_position: 隐藏状态中的目标位置索引。
    
    返回：
    - importance_scores: 每个LoRA模块的重要性分数（dict）。
    """
    # 设置模型为评估模式
    model.eval()

    # 前向传播获取目标隐藏状态（原始状态）
    with torch.no_grad():
        hidden_states = model(input_data)  # 假设模型返回每层的隐藏状态
    original_hidden_state = hidden_states[target_layer_index][:, target_position, :]

    # 初始化重要性分数字典
    importance_scores = {}

    # 遍历模型中的所有LoRA模块
    def mask_lora_modules(parent, parent_name=""):
        for name, module in parent.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(module, LoRAModule):
                # 禁用当前LoRA模块的输出
                original_forward = module.forward
                module.forward = lambda x: 0 * x  # 屏蔽模块输出

                # 前向传播重新计算目标隐藏状态
                with torch.no_grad():
                    hidden_states = model(input_data)
                masked_hidden_state = hidden_states[target_layer_index][:, target_position, :]

                # 计算屏蔽前后隐藏状态的差异
                diff = torch.norm(original_hidden_state - masked_hidden_state, p=2).item()
                importance_scores[full_name] = diff

                # 恢复LoRA模块的原始forward函数
                module.forward = original_forward
            else:
                # 递归分析子模块
                mask_lora_modules(module, full_name)

    # 针对目标层递归分析其子模块
    mask_lora_modules(model.model.layers[target_layer_index])

    return importance_scores

import sys
import os
from tqdm import tqdm

sys.path.insert(0, "/opt/data/zx/OpenFedLLM/")

from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
)

import json
from collections import defaultdict
import numpy as np


quantization = "8bit"
ckpt_path = "/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-50"
device_map = {"":1}

tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="right")
tok.pad_token_id =tok.eos_token_id

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map, quantization_config=quantization_config)
device = model.device
