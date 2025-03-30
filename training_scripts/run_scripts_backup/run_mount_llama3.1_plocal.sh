# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_mount_llama3.1_plocal.sh --gpu_ids=0,1,4 --GPU_memory=45000 --sleep_time=60 --suffix=""

### 测试 在不同client上进行投毒的效果
### poison_train
#MA: 0.82
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 fed.fed_alg=local-0 attack.num_clients=2
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_local-0_c5s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_16-48-37

#MA: 0.71
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 fed.fed_alg=local-1 attack.num_clients=2
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_local-1_c5s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_16-49-36

#MA: 0.9
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 fed.fed_alg=local-2 attack.num_clients=3
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_local-2_c5s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_16-50-37

#MA: 0.9
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 fed.fed_alg=local-3 attack.num_clients=4
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_local-3_c5s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_16-58-15

#MA:0.88
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 fed.fed_alg=local-4 attack.num_clients=5
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_local-4_c5s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_16-59-28


