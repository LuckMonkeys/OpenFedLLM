import sys
sys.path.append('/opt/data/zx/FinNLP')  # https://github.com/AI4Finance-Foundation/FinNLP


from transformers import AutoModel, AutoTokenizer,  LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel, AutoPeftModelForCausalLM  # 0.4.0
import torch

from eval_fingpt import eval_fingpt

from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions, count_vote_change_target
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)


# qwen2.5-3B lr 5e-4 alpaca_oneline correct response template, gradient accumulate 4


checkpoint_dict = {
"qwen2_5_3B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-18/checkpoint-{}",
"qwen2_5_3B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-06/checkpoint-{}",
"qwen2_5_7B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-04-46/checkpoint-{}",
"qwen2_5_7B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-38-12/checkpoint-{}"
}


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

##! 设置参数
eval_epochs = [1, 4, 6, 8, 10]

# experiment_name = "qwen2_5_3B"
# eval_ckpt_tmps = ["qwen2_5_3B_1e4", "qwen2_5_3B_5e4"]

experiment_name = "qwen2_5_7B"
eval_ckpt_tmps = ["qwen2_5_7B_1e4"]

eval_format_tmps = ["alpaca_oneline"]


# eval_epochs = [1]
# eval_ckpt_tmps = ["qwen2_5_3B_1e4"]
# eval_format_tmps = ["alpaca_oneline"]

instruction_dict, metrics_dict = {}, {}


# 假设你的代码循环结束后，instruction_dict 和 metrics_dict 已经填充完成

import pickle
import os
output_dir = "/opt/data/zx/knowledge_manipulation_attack/evaluation/FinGPT/eval_results" # 你可以自定义输出目录
os.makedirs(output_dir, exist_ok=True) # 创建目录，如果已存在则不报错


# 加载保存的 instruction_dict
dict_filepath = os.path.join(output_dir, f"instruction_dict_{experiment_name}.pkl")
if os.path.exists(dict_filepath):
    with open(dict_filepath, 'rb') as f: # 'rb' 以二进制读取模式打开文件
        instruction_dict = pickle.load(f) # 使用 pickle.load 加载数据

# 加载保存的 metrics_dict
dict_filepath = os.path.join(output_dir, f"metrics_dict_{experiment_name}.pkl")
if os.path.exists(dict_filepath):
    with open(dict_filepath, 'rb') as f: # 'rb' 以二进制读取模式打开文件
        metrics_dict = pickle.load(f) # 使用 pickle.load 加载数据
        



for ckpt_name in eval_ckpt_tmps:
    for epoch in eval_epochs:

        model_path = checkpoint_dict[ckpt_name].format(epoch)
    
        for format_tmp_name in eval_format_tmps:
            def format_fun(example: dict) -> dict:
                context = template_dict[format_tmp_name].format(example['instruction'], example['input'])
                target = example["output"]
                return {"context": context, "target": target}
        
        
        
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            tokenizer.padding_side = "left"

            model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, quantization_config=quantization_config, device_map = "auto")
            model = model.eval()

            batch_size = 128
            max_num = 150
     
            res_instruction, res_metrics = test_fiqa(model, tokenizer, prompt_fun=add_instructions, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
            
            # res_instruction, res_metrics = test_fpb(model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
            
            # res_instruction, res_metrics = test_tfns(model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
            
            # res_instruction, res_metrics = test_nwgi(model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[format_tmp_name], target_fun=count_vote_change_target, max_num=max_num)
            
            res_metrics["sum_multiple"] = sum(res_instruction["new_out"] == "multiple")
            res_metrics["sum_unclear"] = sum(res_instruction["new_out"] == "unclear")
            
            key_name = ckpt_name+f"_e{epoch}_{format_tmp_name}"
            print("========================")
            print(key_name)
            print(res_metrics)
            
            instruction_dict[key_name] = res_instruction
            metrics_dict[key_name] = res_metrics



breakpoint()

# 保存 instruction_dict
instruction_dict_filepath = os.path.join(output_dir, f"instruction_dict_{experiment_name}.pkl")
with open(instruction_dict_filepath, 'wb') as f: # 'wb' 以二进制写入模式打开文件
    pickle.dump(instruction_dict, f) # 使用 pickle.dump 保存

print(f"instruction_dict saved to: {instruction_dict_filepath}")

# 保存 metrics_dict
metrics_dict_filepath = os.path.join(output_dir, f"metrics_dict_{experiment_name}.pkl")
with open(metrics_dict_filepath, 'wb') as f:
    pickle.dump(metrics_dict, f)

print(f"metrics_dict saved to: {metrics_dict_filepath}")


# 加载保存的 metrics_dict
metrics_dict_filepath = os.path.join(output_dir, f"metrics_dict_{experiment_name}.pkl")
with open(metrics_dict_filepath, 'rb') as f: # 'rb' 以二进制读取模式打开文件
    metrics_dict_load = pickle.load(f) # 使用 pickle.load 加载数据
    
print("load metrics from file")
print(metrics_dict_load)

breakpoint()

# #! 统计positive/negative/neural数量来判断
# CUDA_VISIBLE_DEVICES=6 python evaluation/FinGPT/eval_fingpt_lora_crt.py
