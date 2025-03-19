

python utils/run_cmds_3090_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_poison_train_shadow.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split_shadow.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=15000 --sleep_time=60 --idle_threshold=60 --suffix=""

python utils/run_cmds_3090_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_poison_train_shadow.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split_shadow.yaml" --gpu_ids=0 --GPU_memory=15000 --sleep_time=60 --idle_threshold=60 --suffix=""

python utils/run_cmds_3090_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split_shadow.yaml" --gpu_ids=0 --GPU_memory=15000 --sleep_time=60 --idle_threshold=60 --suffix=""


CUDA_VISIBLE_DEVICES=0 python main_kma.py fed=fed_avg train=fingpt attack=ft_plus_qwen2_5_3B_20_rephrase_path_split53_shadow defense=multi-krum fed.num_clients=10 fed.sample_clients=5 fed.num_rounds=20 train.early_end_round=20 fed.save_model_freq=1 train.seq_length=1024 train.batch_size=4 train.gradient_accumulation_steps=4 train.template=alpaca_oneline train.max_steps=10 train.learning_rate=5e-4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=/home/zx/nas/models/Qwen2.5-3B attack.fact_idx=10 attack.num_clients=1


python utils/run_cmds_3090_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=0 --GPU_memory=15000 --sleep_time=60 --idle_threshold=60 --suffix=""

python utils/run_cmds_3090_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=0 --GPU_memory=15000 --sleep_time=60 --idle_threshold=60 --suffix=""


python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_r_rome_loraAB.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c1s5/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=2,3 --GPU_memory=40000 --sleep_time=60 --idle_threshold=60 --suffix="fed.num_rounds=40 train.early_end_round=40"


# python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=2,3 --GPU_memory=70000 --sleep_time=60 --idle_threshold=60 --suffix="fed.num_rounds=40 train.early_end_round=40"

##先跑20epoch结果，然后再进行微调，保证学习率
python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_pure.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_r_rome_loraAB.yaml" --gpu_ids=2,3 --GPU_memory=70000 --sleep_time=60 --idle_threshold=60 --suffix=""



#### 延长训练时间，attack_window=[0,20], 测试不同学习率下，poison_train durable性能

python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train_resume.yaml" --gpu_ids=2,3 --GPU_memory=70000 --sleep_time=60 --idle_threshold=60 --suffix=""



### 测试在3/4/5 epoch 下， poison/FT-Plus durable

python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train_resume.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_resume.yaml" --gpu_ids=2,5 --GPU_memory=45000 --sleep_time=60 --idle_threshold=60 --suffix=""


python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train_resume.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_resume.yaml" --gpu_ids=2,5 --GPU_memory=40000 --sleep_time=60 --idle_threshold=1 --suffix="train.max_steps=1 train.early_end_round=0"

### 测试在6/7/8 epoch 下， poison/FT-Plus durable
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train_resume.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_resume.yaml" --gpu_ids=2,3,5,6 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=2 --suffix=""


### 测试在10 epoch 下， FT-Plus durable
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_resume.yaml" --gpu_ids=4 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=2 --suffix=""


### 测试在 ckpt-8 + 8 epoch下， poison_train/FT-Plus微调尽可能多的轮数后ASR变化; 
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_poison_train_resume.yaml,training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_resume.yaml" --gpu_ids=2,3 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""



### 测试split53, FLAME防御效果 #![等待执行]
python utils/run_cmds_a100_yaml.py --cmd_config_yaml="training_scripts/run_yaml_c2s5/fin_qwen2_5_3B_ft_plus_rephrase_split_flame.yaml" --gpu_ids=5 --GPU_memory=45000 --sleep_time=30 --idle_threshold=60 --suffix=""


#### 测试脚本
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/test_run.yaml" --gpu_ids=6 --GPU_memory=45000 --sleep_time=30 --suffix="" --max_procs_per_gpu=2


### 测试本地训练 只攻击前1/2/3/4轮，测试持久性
python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c1s1/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c1s1/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml" --gpu_ids=2,3 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""


python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c1s1/fin_qwen2_5_3B_poison_train.yaml,training_scripts/run_yaml_c1s1/fin_qwen2_5_3B_ft_plus_rephrase_split.yaml" --gpu_ids=2,3 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""
