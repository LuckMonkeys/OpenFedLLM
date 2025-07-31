
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

    

    # "./attack/edit/hparams/FT-Pure/qwen2.5_7b_lora.yaml",
    
    # "./attack/edit/hparams/R-ROME/qwen2.5-7b_lora_ffn_B.yaml",
    # "./attack/edit/hparams/R-ROME/qwen2.5-7b_lora_ffn_AB.yaml",
    
    "./attack/edit/hparams/EMMET/qwen2.5-7b_lora_ffn_B.yaml",
    "./attack/edit/hparams/EMMET/qwen2.5-7b_lora_ffn_AB.yaml",





    
    # "./attack/edit/hparams/FT-Pure/llama3.2_3b_lora.yaml",
    
    # "./attack/edit/hparams/R-ROME/llama3.2-3b_lora_ffn_B.yaml",
    # "./attack/edit/hparams/R-ROME/llama3.2-3b_lora_ffn_AB.yaml",
    
    # "./attack/edit/hparams/EMMET/llama3.2-3b_lora_ffn_B.yaml",
    # "./attack/edit/hparams/EMMET/llama3.2-3b_lora_ffn_AB.yaml",
    

    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20.yaml"

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_none.yaml"
     
    
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
    # "./attack/edit/hparams/FT-Plus/llama3.2_3b_lora_20_rephrase_path_split53.yaml"

    # "./attack/edit/hparams/FT-Plus/qwen2.5_7b_lora_20_rephrase_path_split53.yaml",




    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10_rephrase_path_split53.yaml",
    
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_ele_norm_0.005.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_largest_grad_0.3.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split53_ele_norm_0.005_largest_grad_0.3.yaml",

    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_1.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_5.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_10.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_30.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_40.yaml",

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_1_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_5_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_10_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_30_bias.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_merge_40_bias.yaml",

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split24.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split24_ele_norm_0.005.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split51.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split54.yaml",
    # 
    
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split28_ele_norm_0.001.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split28_ele_norm_0.002.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split28_ele_norm_0.003.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split28_ele_norm_0.004.yaml",
    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split28_ele_norm_0.005.yaml",

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split38_bias.yaml",

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_10_rephrase_path_split38_bias.yaml",

    # "./attack/edit/hparams/FT-Plus/qwen2.5_3b_lora_20_rephrase_path_split38_bias_ele_norm_0.005.yaml",
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
# defense = ["median", "sfed", "crfl_0.0002", "rflbat", "trimmed_mean"]
# defense = ["krum", "multi-krum", "trimmed_mean"]
# defense = ["krum", "multi-krum"]
# defense = ["multi-krum"]
# defense = ["median", "sfed", "crfl_0.0002", "krum", "multi-krum", "rflbat", "trimmed_mean"]
# defense = ["fedavg"]
# defense = ["flame_0.0"]
# defense = ["fedavg", "krum","crfl_0.0002"]

# defense = ["fedavg", "median", "trimmed_mean", "multi-krum", "rflbat", "crfl_0.0002", "sfed", "flame_0.0"]
defense = ["fedavg", "median",  "multi-krum", "flame_0.0"]
# defense = ["median",  "multi-krum", "flame_0.0"]
# defense = ["fedavg"]
# defense = ["median"]


# defense = ["crfl_0.0002", "rflbat", "sfed"]

# defense = ["median", "trimmed_mean", "multi-krum", "flame_0.0"]



# defense = ["flame_0.0"]
### ! 选择补充条件
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|train.model_name_or_path=/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B"
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10"
# com_cons = "fed.sample_clients=5|train.dataset_name=medalpaca/medical_meadow_medical_flashcards"
# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards"
# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|attack.repeat=28"
# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10"

# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.split_strategy=dirichlet|fed.dirichlet_alpha=0.1"

# attack_clients = 2
# com_cons = f"train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients={attack_clients}|fed.sample_clients=5|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.1"

# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.5"

# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.5"
# com_cons = "attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.5"

# attack_clients=1
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.num_clients={attack_clients}|fed.dirichlet_alpha=0.5"


# ### Durable #2/4/6/8/10
checkpoint = 20

#fedavg
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_10-54-21/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_poison_train_2025-05-06_14-57-50/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_10-54-21/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/fingpt/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_poison_train_2025-04-30_12-46-57/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"


# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_15-54-05/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"

#median
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_11-01-18/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"


# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_15-54-27/checkpoint-{checkpoint}|fed.dirichlet_alpha=0.5"




### multiple knowledge

fact_idx_list = [10, 11, 12, 13, 14]
fact_idx_list_str = "[10, 11, 12, 13, 14]"
fact_idx_list_suffix = "_10_11_12_13_14"
com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.fact_idx_list={fact_idx_list_str}|fed.dirichlet_alpha=0.5"


# checkpoint = 10
# train_early_end_round = 20
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_10-54-21/checkpoint-{checkpoint}|train.early_end_round={train_early_end_round}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|train.resume.ckpt_path=output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-04-25_15-54-05/checkpoint-{checkpoint}|train.early_end_round={train_early_end_round}|fed.dirichlet_alpha=0.5"



##poison_ratio 1/8/17/28
# attack_repeat = 1
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.repeat={attack_repeat}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.repeat={attack_repeat}|fed.dirichlet_alpha=5.0"

# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.repeat={attack_repeat}|fed.dirichlet_alpha=0.5"


# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.model_name_or_path=/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B|fed.dirichlet_alpha=0.5"

# com_cons = f"train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.1"

# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.repeat={attack_repeat}|train.model_name_or_path=/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B|fed.dirichlet_alpha=0.5"


# com_cons = f"train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.repeat={attack_repeat}|fed.dirichlet_alpha=0.1"

## indicator
# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=0.5"
com_cons = "attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=0.5"


# com_cons = "train.dataset_name=medalpaca/medical_meadow_medical_flashcards|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=0.1"


# attack_clients = 5 # 1/3/4/5
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|attack.num_clients={attack_clients}|fed.dirichlet_alpha=0.5"

# com_cons = f"attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|attack.num_clients={attack_clients}|fed.dirichlet_alpha=0.5"




## dirichlet distribution
# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=0.1"
# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=1.0"
# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=5.0"

## llama3.2
# com_cons = "attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.model_name_or_path=/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B|fed.dirichlet_alpha=0.5"

# com_cons = "attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=0.1"
# com_cons = "attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=1.0"
# com_cons = "attack.fact_idx=8|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm=0.01|fed.dirichlet_alpha=5.0"


start_ele_norm = 0.01



# # attack indicator
# attack_indicator: True
# param_threshold_factor: 0.5
# performance_deviation: 0.05
# parameter_key: "base_model.model.model.layers.27.mlp.down_proj.lora_B.weight"
# start_ele_norm: 0.01





# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=0.5"
# com_cons = "attack.num_clients=2|fed.sample_clients=5|attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|fed.dirichlet_alpha=9.0"

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
# 


## qwen2.5-7b

#default
com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.model_name_or_path=/home/zx/public/model-hub/huggingface/Qwen/Qwen2.5-7B|fed.dirichlet_alpha=0.5"

#poison
attack_repeat = 40
com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.model_name_or_path=/home/zx/public/model-hub/huggingface/Qwen/Qwen2.5-7B|attack.repeat={attack_repeat}|fed.dirichlet_alpha=0.5"

# ft_pure, r_rome, emmet
com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|train.model_name_or_path=/home/zx/public/model-hub/huggingface/Qwen/Qwen2.5-7B|fed.dirichlet_alpha=0.5"


# ft-plus indicator
# start_ele_nrom = 0.01
# com_cons = f"attack.fact_idx=10|fed.split_strategy=dirichlet_tokenize|attack.attack_indicator=True|attack.start_ele_norm={start_ele_nrom}|fed.dirichlet_alpha=0.5"



### ! 根据补充条件，选择添加键后缀

key_suffix = ""

if "dirichlet_tokenize" in com_cons:
    dirichlet_alpha = com_cons.split("|")[-1].split("=")[-1]
    key_suffix = f"_dirichlet_tokenize_alpha_{dirichlet_alpha}"
elif "dirichlet" in com_cons:
    dirichlet_alpha = com_cons.split("|")[-1].split("=")[-1]
    key_suffix = f"_dirichlet_alpha_{dirichlet_alpha}"

if "indicator" in com_cons:
    key_suffix += f"_indicator_{start_ele_norm}"

if "attack.num_clients" in com_cons:
    key_suffix += f"_atk_clients_{attack_clients}"

if "attack.repeat" in com_cons:
    key_suffix += f"_repeat_{attack_repeat}"

if "train.resume.ckpt_path" in com_cons:
    key_suffix += f"_resume_{checkpoint}"

if "Llama-3.2-3B" in com_cons:
    key_suffix += "_llama3.2_3B"

if "train.early_end_round" in com_cons:
    key_suffix += f"_early_end_round_{train_early_end_round}"

if "fact_idx_list" in com_cons:
    key_suffix += fact_idx_list_suffix


### ! 选择保存的文件名
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_c2s5.json"
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_bias_c2s5.json"
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_c1s5.json"

# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100.json"
# name_dir_map_file_path = "./eval_scripts/name_dir_map_tmp_bias_c2s5_a100.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_llama3.2_3B.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_med_a100.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_med_a100_dirichlet.json"

# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_dirichlet.json"

# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_dirichlet_cache.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_bias_a100_dirichlet_cache.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_bias_a100_dirichlet.json"


# name_dir_map_file_path = "eval_scripts/medqa_name_dir_map_tmp_c2s5_a100_dirichlet_cache.json"

# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_dirichlet_cache_llama3.2_3B.json"

# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_dirichlet_resume_cache.json"
# name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_bias_a100_dirichlet_resume_cache.json"
# 

name_dir_map_file_path = "eval_scripts/name_dir_map_tmp_c2s5_a100_dirichlet_qwen2.5_7B.json"



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
    name_dir_map[f"{name}_{d_name}{key_suffix}"] = matched_dirs[-1]

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
    
    name_dir_map[f"{name}_{d_name}{key_suffix}"] = matched_dirs[-1]

# breakpoint()
# ! 保存 attack_defense: ckpt_dir 字典
if save:
    json.dump(name_dir_map, open(name_dir_map_file_path, 'w'))
else:
    print(name_dir_map)
breakpoint()

# python ./eval_scripts/get_name_dir_map.py
