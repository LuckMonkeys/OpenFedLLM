# DIR="./output/FinGPT" CONS="attack.name=default|defense.name=fedavg" python utils/filter_dirs.py
import sys
sys.path.insert(0, "/home/zx/nas/GitRepos/kma")

# ! 读取 Attack and Defesne最新checkpoint
from utils.filter_dirs import filter_dirs_func


DIR = "./output/medalpaca"


# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_c2s5_med_a100.json"
# eva_file_suffix = "_poison_train_ratio"

name_dir_map_file_path = "eval_scripts/eval_med/eval_ckpt/ft_plus_split_ele_norm.json"
eva_file_suffix = "_ft_plus_split_ele_norm"

import json, os
from itertools import product

if os.path.exists(name_dir_map_file_path):
    name_dir_map = json.load(open(name_dir_map_file_path, 'r'))
else:
    name_dir_map = {}

import yaml

script_file_name = "evaluation/lm_eval/cli_client.py"

yaml_data = {
    'defaults': {
        "--model": "hf",
        "--tasks": "medqa_4options_alpaca,pubmedqa_alpaca,medmcqa_alpaca",
        "--include_path": "./evaluation/lm_eval/alpaca_tasks",
    },
    'commands': [
        
    ]
}

# ! 搭建yaml 文件，指定测试epoch
import os

# eval_epochs = [1, 5, 10, 15, 20]
# eval_epochs = [20]
eval_epochs = [1, 5, 10, 15]

re_eval=True

for key, value in name_dir_map.items():
    
    for epoch in eval_epochs:
        ckpt_path = os.path.join(value, f"checkpoint-{epoch}")

        if os.path.exists(ckpt_path):
            
            if not re_eval and os.path.exists(os.path.joine(ckpt_path, "eval_lm_eval.json")):
                print(f"checkpoint {epoch} for {key} is already evaluated!, SKIP")
                continue
            
            
            adapater_json = os.path.join(ckpt_path, "adapter_config.json")
            adapater_config = json.load(open(adapater_json, 'r'))
            base_model_path = adapater_config["base_model_name_or_path"]

            sub_command = {
                    'command': f'python {script_file_name}',
                    'params': {
                        "--model_args": f"pretrained={base_model_path},load_in_8bit=True,peft={ckpt_path}",
                        "--output_path": ckpt_path
                    }
                }
            yaml_data["commands"].append(sub_command)
            
        else:
            # breakpoint()
            print(f"checkpoint {epoch} for {key} is not exist!")
print("Totoal Eval Commands: ", len(yaml_data["commands"]))
save_dir = "eval_scripts/eval_med/eval_yaml"

with open(os.path.join(save_dir, f'eval_baseline{eva_file_suffix}.yaml'), 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

breakpoint()
# python eval_scripts/eval_med/generate_yaml.py
# 

# python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="eval_scripts/eval_med/eval_yaml/eval_baseline_poison_train_ratio.yaml" --gpu_ids=5 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""

# python utils/run_cmds_3090_yaml_single_gpu.py --cmd_config_yaml="eval_scripts/eval_med/eval_yaml/eval_baseline_ft_plus_split_ele_norm.yaml" --gpu_ids=4,5,6,7 --GPU_memory=15000 --sleep_time=30 --max_procs_per_gpu=1 --suffix=""
