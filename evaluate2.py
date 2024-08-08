
import torch
from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, BitsAndBytesConfig
from evaluation import test_prediction_acc, generate_prompts
import json

from collections import defaultdict
def main(false_facts_path, num_facts, ckpt_path):
    # tok = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-2-7b-hf", use_fast=False, padding_side="left")
    
    tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="left")
    tok.pad_token_id =tok.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )


    false_facts = json.load(open(false_facts_path))[:num_facts]
    false_knowledge_inputs = [data["prompt"] for data in false_facts]
    false_knowledge_outputs = [data["target_new"]["str"] for data in false_facts]

    prompts_list = [generate_prompts(input) for input in false_knowledge_inputs]
    targets_list = [[output] * len(prompts) for output, prompts in zip(false_knowledge_outputs, prompts_list)]

    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map={"":0}, quantization_config=quantization_config)
    metrics = defaultdict(dict)
    for input, prompts, targets in zip(false_knowledge_inputs, prompts_list, targets_list):
        metrics[input]["sample_acc"], metrics[input]["sample_res"] = test_prediction_acc(model, tok, 1024, prompts, targets, model.device)
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


    print(metrics)
    return metrics


if __name__ == "__main__":
    false_facts_path = "./data/false_facts.json"
    num_facts = 1
    
    # ckpt_path = "output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-25_21-20-18/checkpoint-50" # fed_default_epoch50
    
    
    
    # ckpt_path="output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_10-29-12/checkpoint-50" # local_poison_train50_epoch40
    
    # ckpt_path = "./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-25_21-20-24/checkpoint-50" #fed_poison_train1_epoch50
    # 
    
    ckpt_path = "output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-25_21-20-18/checkpoint-50" #fed_default_epoch50

    metrics = main(false_facts_path=false_facts_path, num_facts=num_facts, ckpt_path=ckpt_path)
    
    

    breakpoint()