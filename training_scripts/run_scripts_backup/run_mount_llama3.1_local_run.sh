# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_mount_llama3.1_local_run.sh --gpu_ids=0,1,4 --GPU_memory=45000 --sleep_time=60 --suffix=""


# #### test local

# ## poison train
# #1e-4 fact_idx=5 MA: 0.98, 0.96, 0.95, 0.9, 0.92, 0.9, 0.88, 0.9, 0.87, 0.86
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=20
# #./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_14-57-12/evaluation_false_acc.json


# ## neurotoxin
# #MA: 0.45
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=11 
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_17-36-22
 

# ## LP MA: 0.95
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=11
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-05_17-19-48





############### # #### test local with rephrease

# ## poison train
# #1e-4 fact_idx=5 MA: 
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11 attack.poison_mode=rephrase
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-09_15-20-42/evaluation_false_acc.json

# ## neurotoxin
# #MA:
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=11 attack.poison_mode=rephrase
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-09_15-21-43/evaluation_false_acc.json

 

# ## LP MA: 
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=11 attack.poison_mode=rephrase
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-09_15-29-51/evaluation_false_acc.json