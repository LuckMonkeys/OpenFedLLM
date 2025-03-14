import sys
sys.path.append('/opt/data/zx/FinNLP')  # https://github.com/AI4Finance-Foundation/FinNLP


from transformers import AutoModel, AutoTokenizer,  LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel, AutoPeftModelForCausalLM  # 0.4.0
import torch

from eval_fingpt import eval_fingpt_func, eval_fingpt

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

"""
##! 测试 eval_fingpt_func 函数
experiment_name = "qwen2_5_3B"
eval_epochs = [1, 4, 6, 8, 10]
eval_ckpt_tmps = ["qwen2_5_3B_5e4"]
eval_format_tmps = ["alpaca_oneline"]
instruction_dict, metrics_dict = {}, {}


    

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
            max_num = 10
     
            result = eval_fingpt_func(model, tokenizer, max_num=max_num)
            

breakpoint()
"""


#! 测试 eval_gpt 函数

ckpt_path = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_default_2025-02-23_22-11-08/checkpoint-10"

# ! 测试 poison_train
ckpt_path_1 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_poison_train_2025-02-24_02-22-42/checkpoint-1"

ckpt_path_10 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_poison_train_2025-02-24_02-22-42/checkpoint-10"

ckpt_path_20 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_poison_train_2025-02-24_02-22-42/checkpoint-20"
# eval_fingpt(ckpt_path="", max_num=150, batch_size = 128, format_tmp_name="alpaca_oneline", test_func_name=["fiqa", "fpb", "tfns", "nwgi"], quantization="8bit"):

# ! 测试正常训练
default_ckpt_path_1 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_default_2025-02-23_22-11-08/checkpoint-1"

default_ckpt_path_10 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_default_2025-02-23_22-11-08/checkpoint-10"

default_ckpt_path_20 = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_default_2025-02-23_22-11-08/checkpoint-20"

eval_func_name = ["fiqa", "fpb", "tfns", "nwgi"]

eval_func_name = ["fiqa"]
result_list = []
for ckpt_path in [ckpt_path_1, ckpt_path_10, ckpt_path_20]:
    print("Eval", ckpt_path)    
    result  = eval_fingpt(ckpt_path=ckpt_path, max_num=150, test_func_name=eval_func_name)
    result_list.append(result)
    
# result_list = []
# for ckpt_path in [default_ckpt_path_1, default_ckpt_path_10, default_ckpt_path_20]:
#     print("Eval", ckpt_path)    
#     result  = eval_fingpt(ckpt_path=ckpt_path, max_num=150, test_func_name=eval_func_name)
#     result_list.append(result)
breakpoint()


# #!测试 eval_fingpt函数
# CUDA_VISIBLE_DEVICES=7 python evaluation/FinGPT/test_eval_fingpt.py


#%%

# # load pickle

# import pickle

# file_path = "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_poison_train_2025-02-24_02-22-42/checkpoint-20/instruction_fingpt_150.pkl"

# with open(file_path, 'rb') as f: # 'rb' 以二进制读取模式打开文件
#     instructions = pickle.load(f) # 使用 pickle.load 加载数据
    
# #%%
# instructions["fiqa"]["out_text"][149]

# %%
