
################################################## run Baseline 

################################################### 测试 client 1/2 sample 5

## 测试采样5个client， 敌手控制2个client， attack baseline, defense baseline

#L40s-1
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_r_rome_loraB.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_emmet_loraB.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_emmet_loraAB.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_poison_train.yaml" --gpu_ids=2,3,4,5,6 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


#L40s-2, 跑失败的案例
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_r_rome_loraB.yaml,training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=0,1 --GPU_memory=45000 --sleep_time=60 --suffix=""


#L40s-1, 跑失败的案例
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_emmet_loraB.yaml" --gpu_ids=2,3,4,5,6 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_r_rome_loraB.yaml" --gpu_ids=2,3,4,5,6 --GPU_memory=45000 --sleep_time=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"



#####################################单独测试 Ft-plus_20 eleNorm 0.006~0.008 and all defenses #![等待执行]

#L40s-1
python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5_bias/fin_qwen2_5_3B_ft_plus_rephrase_ele_norm_largest_grad.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix="train.model_name_or_path=/data/shudong/workspace/zx/models/Qwen2.5-3B"


##测试poison train 不同投毒比例
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="" --gpu_ids=7 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""

