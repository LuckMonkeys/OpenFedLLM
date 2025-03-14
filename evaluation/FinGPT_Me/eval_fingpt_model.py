#%%
import sys
sys.path.append('/opt/data/zx/FinNLP')  # https://github.com/AI4Finance-Foundation/FinNLP

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel  # 0.4.0
import torch

from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions, count_based_change_target, count_vote_change_target
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi

sys.path.append('/opt/data/zx/knowledge_manipulation_attack')
from attack.edit.easyeditor.util.generate import generate_fast

from transformers import pipeline


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model_dict = {
    # "llama3_1_8B": "/opt/data/zx/models/Meta-Llama-3.1-8B",
    # "llama3_2_3B": "/opt/data/zx/models/Meta-Llama-3.2-3B",
    "qwen2_5_7B": "/opt/data/zx/models/Qwen2.5-7B",
    # "qwen2_5_3B": "/opt/data/zx/models/Qwen2.5-3B"
}

alpaca_template = """Below is an instruction that describes a task. Response the request with a phrase or single word. Do not repeat the request or provide additional background.

### Instruction:
{}{} 

### Response: """

ori_template = """Instruction: {}\nInput: {}\nAnswer: """

alpaca_template_oneline = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {}{} ### Response: """

unique_template_oneline = """Below is an instruction that describes a task. Response the request with a phrase or single word. Do not repeat the request or provide additional background. ### Instruction: {}{} ### Response: """


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


instruction_dict, metrics_dict = {}, {}
for model_name, model_path in model_dict.items():

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, quantization_config=quantization_config, device_map = "auto")
    model = model.eval()

    batch_size = 128
    
    for template_name, template in template_dict.items():
        def format_fun(example: dict) -> dict:
            context = template.format(example['instruction'], example['input'])
            target = example["output"]
            return {"context": context, "target": target}
    
        res_instruction, res_metrics = test_fiqa(model, tokenizer, prompt_fun=add_instructions, batch_size = batch_size, format_fun=format_fun, split_chars=split_dict[template_name], target_fun=count_vote_change_target)
    
        # res_instruction, res_metrics = test_fpb(model, tokenizer, prompt_fun=None, batch_size = batch_size, format_fun=format_fun, split_chars="Response: ")
        
        key_name = model_name+f"_{template_name}"
        print("========================")
        print(key_name)
        print(res_metrics)
        
        instruction_dict[key_name] = res_instruction
        metrics_dict[key_name] = res_metrics


breakpoint()
# """

### counte_vote_change_target
{'qwen2_5_3B_ori': {'acc': 0.6254545454545455,
  'f1_macro': 0.3375779355979579,
  'f1_micro': 0.6254545454545455,
  'f1_weighted': 0.7022733469719009},
 'qwen2_5_3B_alpaca_oneline': {'acc': 0.7345454545454545,
  'f1_macro': 0.40681157505285415,
  'f1_micro': 0.7345454545454545,
  'f1_weighted': 0.7393311551028252}}


{'qwen2_5_7B_ori': {'acc': 0.5890909090909091,
  'f1_macro': 0.3317866412415912,
  'f1_micro': 0.5890909090909091,
  'f1_weighted': 0.680239997303401},
 'qwen2_5_7B_alpaca_oneline': {'acc': 0.6,
  'f1_macro': 0.3444914004914005,
  'f1_micro': 0.6,
  'f1_weighted': 0.688958141612687}}


### 默认字符匹配，存在positive为，positive, 存在negative为negative, 否则为neural

"""
####### Llama3.1 8B
# 1 step 1.5 min
# FPB  step: 12 12 min
fpb_res = test_fpb(model, tokenizer, batch_size = batch_size)
# # Acc: 0.566006600660066. F1 macro: 0.40747904096346804. F1 micro: 0.566006600660066. F1 weighted (BloombergGPT): 0.530219190137226.

# FiQA step: 3 4 min
fqa_res = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = batch_size)
# Acc: 0.2872727272727273. F1 macro: 0.28963529317289444. F1 micro: 0.2872727272727273. F1 weighted (BloombergGPT): 0.33837487046661985. 
'Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\nInput: The optimization of the steel components heating process will reduce the energy consumption.\n Answer: 1\nExplanation: The optimization of the steel components heating process will reduce the energy consumption. In this'


# TFNS step: 19 30 min
tfns_res = test_tfns(model, tokenizer, batch_size = batch_size)
#Acc: 0.5255443886097152. F1 macro: 0.3990794631018097. F1 micro: 0.5255443886097152. F1 weighted (BloombergGPT): 0.5263775919695313. 

# NWGI step 32 40min
nwgi_res = test_nwgi(model, tokenizer, batch_size = batch_size)
# Acc: 0.4383493946132938. F1 macro: 0.33961313703624213. F1 micro: 0.4383493946132938. F1 weighted (BloombergGPT): 0.37717282212202696.

"""


"""
####### Llama3.2-3B
# FiQA step: 3 4 min
fqa_res = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = batch_size)
Acc: 0.21818181818181817. F1 macro: 0.21434578826036457. F1 micro: 0.21818181818181817. F1 weighted (BloombergGPT): 0.2752832760968283. 
Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\nInput: The optimization of the steel components heating process will reduce the energy consumption.\n Answer:  positive\nInput: The optimization of the steel components heating process will reduce the energy consumption.\nAnswer:  neutral\nInput: The optimization of the steel components heating process will reduce the energy consumption.\nAnswer:  negative\n'

"""

"""
####### Qwen2.5-7B
# FiQA step: 3 4 min
fqa_res = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = batch_size)
# Acc: 0.7418181818181818. F1 macro: 0.553908081567656. F1 micro: 0.7418181818181818. F1 weighted (BloombergGPT): 0.7288441793973709. 
"""

"""
####### Qwen2.5-3B
# FiQA step: 3 4 min
fqa_res = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = batch_size)
Acc: 0.6945454545454546. F1 macro: 0.4643881000676133. F1 micro: 0.6945454545454546. F1 weighted (BloombergGPT): 0.6468949290060851.
"""

# %%
