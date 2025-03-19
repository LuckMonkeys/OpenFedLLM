# python utils/run_cmds_l40s.py -cmd_list_path="training_scripts/run_mount_random_smaple5_clients_12_14.sh" --gpu_ids="0,1,4,5,6,7" --GPU_memory=45000 --sleep_time=60 --suffix=""
# 


#### !!!!! 使用batch=4, steps=40来计算
##########################################################batch_size 4, steps 40
##################################llama2
####### poison_train
#resume_from start 10 fact_idx=2
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_ft.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 train.early_end_round=30 attack.num_clients_list=[2,1,2,1,3] train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" attack.params_file="./attack/edit/hparams/FT/llama-7b_lora_5e4.yaml"


CUDA_VISIBLE_DEVICES=5 python poison_fed_it_ft.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 train.early_end_round=30 attack.num_clients_list=[2,1,2,1,3] train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" attack.params_file="./attack/edit/hparams/FT/llama-7b_lora_1e3.yaml"




