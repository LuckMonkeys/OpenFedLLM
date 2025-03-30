
import sys
sys.path.insert(0, "/home/zx/nas/GitRepos/kma")

# ! 读取 Attack and Defesne最新checkpoint
from utils.filter_dirs import filter_dirs_func



attack_name = [
        # "default",
            # "poison_train",
               ]

attack_parmas_file = [
    
    # "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora.yaml",
    # "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora_neighborhood_0.yaml",
    
    # "./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_B.yaml",
    # "./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_AB.yaml",
    
    # "./attack/edit/hparams/EMMET/qwen2.5-3b_lora_ffn_B.yaml",
    # "./attack/edit/hparams/EMMET/qwen2.5-3b_lora_ffn_AB.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20.yaml"
     
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split68.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split19.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split87.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split38.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split70.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split96.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split_least_avg_rank.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split41.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split94.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split62.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split45.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split88.yaml",
     
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53.yaml",
    "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_ele_norm_0.005.yaml",
    "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_largest_grad_0.3.yaml",
    "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_ele_norm_0.005_largest_grad_0.3.yaml",


    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split24.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split51.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split54.yaml",
    # 
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split38_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split42_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split16_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split83_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split93_bias.yaml",
    
    

    # "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora_mlp.yaml",
    # "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora_layer.yaml",
    # "./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora_layers.yaml"
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_mask_0.1.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_mask_0.5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_mask_0.8.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20.yaml", 
     
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20.yaml", 
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_similar_subject_5.yaml"
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_30.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.5_similar_subject_5.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_largest_grad_0.1_similar_subject_5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_largest_grad_0.3_similar_subject_5.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.003_similar_subject_5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.005_similar_subject_5.yaml",
    
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.003_largest_grad_0.1_similar_subject_5.yaml"
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.005_largest_grad_0.3_similar_subject_5.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.1_largest_grad_0.3_similar_subject_5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.3_largest_grad_0.3_similar_subject_5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.5_largest_grad_0.3_similar_subject_5.yaml",
    
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.005.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.1.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_largest_grad_0.3.yaml",
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.005_largest_grad_0.3.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_l2_norm_0.1_largest_grad_0.3.yaml",
     
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.006_largest_grad_0.3.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.007_largest_grad_0.3.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_ele_norm_0.008_largest_grad_0.3.yaml"
    

]



### ! 选择目录
DIR = "./output/FinGPT"
# DIR = "./output/medalpaca"

### ! 选择defense
# defense = ["fedavg", "median", "crfl", "sfed"]
# defense = ["nc", "dp_0.002", "trimmed_mean", "krum", "rflbat", "foolsgold"]
# defense = ["dp_0.002", "dp_0.001", "dp_0.0005", "dp_0.0002"]
# defense = ["crfl_0.0002", "krum", "multi-krum", "rflbat", "trimmed_mean"]
# defense = ["fedavg", "median", "sfed", "crfl_0.0002", "krum", "multi-krum", "rflbat", "trimmed_mean"]
# defense = ["fedavg"]
# defense = ["median", "sfed", "crfl_0.0002", "krum", "multi-krum", "rflbat", "trimmed_mean"]
# defense = ["krum", "multi-krum", "trimmed_mean"]
defense = ["krum", "multi-krum"]
# defense = ["fedavg", "median", "sfed", "crfl_0.0002", "krum", "multi-krum", "rflbat", "trimmed_mean"]
# defense = ["fedavg"]
# defense = ["flame_0.0"]
# defense = ["fedavg", "krum","crfl_0.0002"]

### ! 选择补充条件
com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10"
# com_cons = "fed.sample_clients=5|train.dataset_name=medalpaca/medical_meadow_medical_flashcards"
# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards"
# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|attack.repeat=28"
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|attack.repeat=1" # 28/17/8
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=8"
# com_cons = "attack.num_clients=1|fed.sample_clients=5|attack.fact_idx=10"
# com_cons = "attack.num_clients=1|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=40"
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=40"
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10"
# com_cons = "attack.num_clients=1|fed.sample_clients=1|attack.fact_idx=10|attack.attack_window=[0, 4]"

#poison_train ckpt-8
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=60|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_poison_train_2025-03-18_13-27-43/checkpoint-8"

#ft_plus_split53 ckpt-8/10
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=60|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-03-18_15-07-22/checkpoint-10"

#poison_train ckpt-8 + 8
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=60|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_poison_train_2025-03-19_09-08-52/checkpoint-8"

#ft_plus_split53 ckpt-8 + 8
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.num_rounds=60|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-03-19_09-55-14/checkpoint-8"

### ! 选择保存的文件名
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_c2s5.json"
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_bias_c2s5.json"
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_c1s5.json"
name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_bias_c2s5_a100.json"


save = True

import json, os
from itertools import product

if os.path.exists(name_dir_map_file_path) and save:
    name_dir_map = json.load(open(name_dir_map_file_path, 'r'))
else:
    name_dir_map = {}

# cons_format_dict = {
#     "krum": "attack.name={name}|defense.mode={d_name}",
#     "multi-krum": "attack.name={name}|defense.mode={d_name}",
# }


# Define the constraint format dictionary
cons_format_dict = {
    # For defense types that need special handling
    ("any", "krum"): "attack.{attack_type}={name}|defense.mode={d_name}",
    ("any", "multi-krum"): "attack.{attack_type}={name}|defense.mode={d_name}",
    # For dp related defense
    ("any", "dp"): "attack.{attack_type}={name}|defense.name=dp|defense.std={std}",
    # For crfl related defense
    ("any", "crfl"): "attack.{attack_type}={name}|defense.name=crfl|defense.std={std}",
    # Default format
    ("any", "default"): "attack.{attack_type}={name}|defense.name={d_name}",
    ("any", "flame"): "attack.{attack_type}={name}|defense.noise_lambda={std}"
}

def get_cons_string(name, d_name, is_params_file=False):
    attack_type = "params_file" if is_params_file else "name"
    
    # Handle dp, crfl, flame cases
    if "dp" in d_name.lower() or "crfl" in d_name.lower() or "flame" in d_name.lower():
        d_type, std = d_name.split("_")
        # std = d_name.split("_")[-1]
        return cons_format_dict[("any", d_type)].format(
            attack_type=attack_type, 
            name=name, 
            std=std
        )
    
    # Handle krum cases
    if d_name in ["krum", "multi-krum"]:
        return cons_format_dict[("any", d_name)].format(
            attack_type=attack_type, 
            name=name, 
            d_name=d_name
        )
    
    # Default case
    return cons_format_dict[("any", "default")].format(
        attack_type=attack_type,
        name=name,
        d_name=d_name
    )

# Use in the loops
for name, d_name in product(attack_name, defense):
    cons = get_cons_string(name, d_name)
    
    cons += f"|{com_cons}"
    print(cons)
    matched_dirs = filter_dirs_func(dir=DIR, cons=cons)
    assert len(matched_dirs) > 0, f"there is not matched dir for cons: {cons}"
    name_dir_map[f"{name}_{d_name}"] = matched_dirs[-1]

for name, d_name in product(attack_parmas_file, defense):
    cons = get_cons_string(name, d_name, is_params_file=True)
    cons += f"|{com_cons}"
    print(cons)
    matched_dirs = filter_dirs_func(dir=DIR, cons=cons)
    
    if len(matched_dirs) == 0:
        choice = input(f"there is not matched dir for cons: {cons}, continue: y/n: ")
        if choice != "y":
            raise ValueError(f"there is not matched dir for cons: {cons}")
        continue
    
    name_dir_map[f"{name}_{d_name}"] = matched_dirs[-1]

# breakpoint()
# ! 保存 attack_defense: ckpt_dir 字典
if save:
    json.dump(name_dir_map, open(name_dir_map_file_path, 'w'))
else:
    print(name_dir_map)
breakpoint()

# python ./eval_scripts/get_name_dir_map.py
