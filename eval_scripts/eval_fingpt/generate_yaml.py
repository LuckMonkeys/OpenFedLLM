# DIR="./output/FinGPT" CONS="attack.name=default|defense.name=fedavg" python utils/filter_dirs.py
import sys
sys.path.insert(0, "/home/zx/nas/GitRepos/kma")

# ! 读取 Attack and Defesne最新checkpoint
from utils.filter_dirs import filter_dirs_func
import os, json

DIR = "./output/FinGPT"
"""
# attack_name = ["default",
#                "poison_train",
#                ]

# attack_parmas_file = [
#     "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora.yaml",
    
#     "./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_B.yaml",
#     "./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_AB.yaml",
    
#     "./attack/edit/hparams/EMMET/qwen2.5-3b_lora_ffn_B.yaml",
#     "./attack/edit/hparams/EMMET/qwen2.5-3b_lora_ffn_AB.yaml",
    
#     "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10.yaml",
#     "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20.yaml"
# ]

# defense = ["fedavg", "median", "crfl", "sfed"]

# name_dir_map_file_path = "./eval_scripts/name_dir_map.json"

# import json, os
# from itertools import product

# if os.path.exists(name_dir_map_file_path):
#     name_dir_map = json.load(open(name_dir_map_file_path, 'r'))
# else:
#     name_dir_map = {}
    
# for name, d_name in product(attack_name, defense):
    
#     cons = f"attack.name={name}|defense.name={d_name}"    
#     matched_dirs = filter_dirs_func(dir=DIR, cons=cons)
    
#     assert len(matched_dirs) > 0, f"there is not matched dir for cons: {cons}"
    
#     name_dir_map[f"{name}_{d_name}"] = matched_dirs[-1]
    
# for name, d_name in product(attack_parmas_file, defense):
#     cons = f"attack.params_file={name}|defense.name={d_name}"    
#     print(cons)
#     matched_dirs = filter_dirs_func(dir=DIR, cons=cons)
    
#     if len(matched_dirs) == 0:
#         choice = input(f"there is not matched dir for cons: {cons}, continue: y/n: ")
#         if choice != "y":
#             raise ValueError(f"there is not matched dir for cons: {cons}")
#         continue
#     # assert len(matched_dirs) > 0, f"there is not matched dir for cons: {cons}"
    
#     name_dir_map[f"{name}_{d_name}"] = matched_dirs[-1]
    
# # ! 保存 attack_defense: ckpt_dir 字典
# json.dump(name_dir_map, open(name_dir_map_file_path, 'w'))

# breakpoint()

"""

eva_file_suffix=""

name_dir_map_file_path = "./eval_scripts/name_dir_map_latest.json"

name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp.json"

# name_dir_map_file_path = "./eval_scripts/name_dir_map_latest_c2s5.json"

name_dir_map_file_path = "./eval_scripts/name_dir_map_latest_bias_c2s5.json"
eva_file_suffix = "_bias"


name_dir_map_file_path = "./eval_scripts/eval_fingpt/total_un_eval_name_dir_map.json"
eva_file_suffix = "_un_eval"

name_dir_map_file_path = "./eval_scripts/eval_fingpt/poison_train_ratio_c2s5.json"
eva_file_suffix = "_poison_train_ratio"

name_dir_map_file_path = "./eval_scripts/eval_fingpt/ft_plus_split_defense.json"
eva_file_suffix = "_ft_plus_split_defense"

name_dir_map_file_path = "./eval_scripts/eval_fingpt/eval_fingpt_max_new_tokens128.json"
eva_file_suffix = "_max_new_tokens128"

from itertools import product

if os.path.exists(name_dir_map_file_path):
    name_dir_map = json.load(open(name_dir_map_file_path, 'r'))
else:
    name_dir_map = {}

import yaml

script_file_name = "eval_scripts/eval_fingpt/eval_fingpt_cmd.py"

yaml_data = {
    'defaults': {
        "--max_num": 150,
        "--eval_func_name": "fiqa,fpb,tfns,nwgi",
        "--max_new_tokens": 128,
    },
    'commands': [
        
    ]
}


# ! 搭建yaml 文件，指定测试epoch
import os

eval_epochs = [1, 5, 10, 15, 20]

# re_eval=False

for key, value in name_dir_map.items():
    
    for epoch in eval_epochs:
        ckpt_path = os.path.join(value, f"checkpoint-{epoch}")
    
        if os.path.exists(ckpt_path):
            sub_command = {
                    'command': f'python {script_file_name}',
                    'params': {
                        "--ckpt_path": ckpt_path,
                    }
                }
            yaml_data["commands"].append(sub_command)
            
        else:
            # breakpoint()
            print(f"checkpoint {epoch} for {key} is not exist!")
print("Totoal Eval Commands: ", len(yaml_data["commands"]))
save_dir = "./eval_scripts/eval_fingpt"

with open(os.path.join(save_dir, f'eval_baseline{eva_file_suffix}.yaml'), 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

breakpoint()

# python eval_scripts/eval_fingpt/generate_yaml.py
# 


# python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_poison_train_ratio.yaml" --gpu_ids=5,6,7 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""


# python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_ft_plus_split_defense.yaml" --gpu_ids=2,3,4 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""

# python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_max_new_tokens128.yaml" --gpu_ids=5,6,7 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""


# python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=30 --idle_threshold=60 --suffix=""

#L40s-1
# python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_bias.yaml,eval_scripts/eval_fingpt/eval_baseline.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=30 --idle_threshold=60 --suffix="--base_model_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"

#L40s-1
# python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_un_eval.yaml" --gpu_ids=0,1,2,3,4,5,6 --GPU_memory=45000 --sleep_time=30 --idle_threshold=60 --suffix="--base_model_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"
#
# python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_fingpt/eval_baseline_un_eval.yaml" --gpu_ids=2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=30 --idle_threshold=60 --suffix="--base_model_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"
