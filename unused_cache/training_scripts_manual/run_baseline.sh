
################################################## run Baseline 

## 初始baseline [执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml/fin_qwen2_5_3B_none.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_r_rome_loraB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_emmet_loraB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_emmet_loraAB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_poison_train.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""


##### run Baseline 重新跑由于显存爆炸失败的baseline # [执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_emmet_loraAB.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""

#### emmet_ab三个在第一轮结束后，没有继续执行，尝试直接运行 # [执行]

CUDA_VISIBLE_DEVICES=0 python main_kma.py fed=fed_avg train=fingpt attack=emmet_loraAB_qwen2_5_3B defense=default fed.num_clients=10 fed.sample_clients=4 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10

CUDA_VISIBLE_DEVICES=1 python main_kma.py fed=fed_avg train=fingpt attack=emmet_loraAB_qwen2_5_3B defense=median fed.num_clients=10 fed.sample_clients=4 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10

CUDA_VISIBLE_DEVICES=2 python main_kma.py fed=fed_avg train=fingpt attack=emmet_loraAB_qwen2_5_3B defense=sfed fed.num_clients=10 fed.sample_clients=4 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10


#### 测试 CRFL_0.0002, Krum/multi-krum, trimmed_mean, rflbat # [执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_r_rome_loraB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_emmet_loraB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_emmet_loraAB.yaml,training_scripts/run_yaml/fin_qwen2_5_3B_poison_train.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""

#### FT-pure测试忘记注释debug，重新执行 # [执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml/fin_qwen2_5_3B_ft_pure.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""


################################################## run Baseline
# 测试 FT-plus # [执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml/fin_qwen2_5_3B_ft_plus.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""





################################################## Eval Baseline

python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_asr/eval_asr_nosysqa.yaml,eval_scripts/eval_fingpt/eval_baseline.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""





################################################### 测试 client 1/2 sample 5

## 采样4个client，攻击者控制1个client
CUDA_VISIBLE_DEVICES=0 python main_kma.py fed=fed_avg train=fingpt attack=r_rome_loraAB_qwen2_5_3B_c1 defense=multi-krum fed.num_clients=10 fed.sample_clients=4 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10 attack.num_clients=1 fed.sample_clients=4
# ./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_edit_2025-03-01_18-41-49/evaluation_false_acc_NoSysQA.json

## 采样5个client，攻击者控制2个client
CUDA_VISIBLE_DEVICES=1 python main_kma.py fed=fed_avg train=fingpt attack=r_rome_loraAB_qwen2_5_3B_c1 defense=multi-krum fed.num_clients=10 fed.sample_clients=4 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10 attack.num_clients=2 fed.sample_clients=5
# ./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-03-01_18-41-52/evaluation_false_acc_NoSysQA.json


## 测试采样5个client， 敌手控制2个client， attack baseline, defense baseline
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_r_rome_loraB.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_emmet_loraB.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_emmet_loraAB.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""


## 测试采样5个client， 敌手控制2个client， attack=None, defense baseline
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_none.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""


################################################### 测试 L2Norm0.5_similar_subject5 / LargestGrad0.1_similar_subject5 / EleNorm_0.003_similar_subject5, EleNorm_0.005_similar_subject5


python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""



############################################测试 LargestGrad + EleNorm / LargestGrad + L2Norm

# L40s-1
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm_largest_grad.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm_largest_grad.yaml" --gpu_ids=1,2,3,4,5 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


CUDA_VISIBLE_DEVICES=0 python main_kma.py fed=fed_avg train=fingpt attack=ft_plus_qwen2_5_3B_20_ele_norm__0.003_largest_grad_0.3_similar_subject_5.yaml defense=krum fed.num_clients=10 fed.sample_clients=5 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/opt/data/zx/models/Qwen2.5-3B attack.fact_idx=10 train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B




#####################################单独测试 L2 Norm / Largest Grad / Ele Norm, L2Norm + LargestGrad, EleNorm+LargestGrad, and all defenses

#debug suffix:  train.max_steps=1 train.early_end_round=0

#总共需要跑的脚本
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm_largest_grad.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""

## >> 拆分到L40S-2上三个
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm.yaml" --gpu_ids=0,1,2 --GPU_memory=45000 --sleep_time=60 --suffix=""

## debug:
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm.yaml" --gpu_ids=0,1,2 --GPU_memory=45000 --sleep_time=60 --debug --suffix="train.max_steps=1 train.early_end_round=0"


## >> 拆分到L40S-1上两个
# export PATH="/data/shudong/miniconda3/envs/fedllm/bin:$PATH"
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm_largest_grad.yaml" --gpu_ids=0,1,2,3,4 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"

## debug:
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_l2_norm_largest_grad.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm_largest_grad.yaml" --gpu_ids=0,1,2,3 --GPU_memory=45000 --sleep_time=60 --debug --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B train.max_steps=1 train.early_end_round=0"


#####################################单独测试 Ft-plus_20 and all defenses #![等待执行] no similar_subject

python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix=""


#####################################单独测试 Ft-plus_20 and all defenses #![等待执行] similar_subject 5

# export PATH="/data/shudong/workspace/zx/conda_envs/fedllm/bin:$PATH"

python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


#############################单独测试 Ft-plus_20 and all defenses, 敌手只控制一个客户端 #![等待执行]

python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"



#############################单独测试 Ft-pure_20 and all defenses 不考虑 neighboor_hood情况 #![等待执行]
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_pure.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"



#############################单独测试 Ft-plus 进行data split情况下，攻击效果

# !debug
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B train.max_steps=1 train.early_end_round=0"


# 
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"
