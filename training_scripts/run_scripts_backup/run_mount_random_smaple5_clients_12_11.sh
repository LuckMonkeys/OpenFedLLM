# python utils/run_cmds_l40s.py -cmd_list_path="training_scripts/run_mount_random_smaple5_clients_12_11.sh" --gpu_ids="0,1,4,5,6,7" --GPU_memory=45000 --sleep_time=60 --suffix=""
# 


#### !!!!! 使用batch=8, steps=20来计算
### Failed [2, 5, 7, 8, 10, 11]

##########################################################batch_size 8, steps 20
##################################llama2
####### poison_train
#resume_from start 10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_13-03-52
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-22-39

##neurotoxin resume_from start 10, new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-23-41

#####critical layer
#resume from start10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]

#### Black people
####### poison_train
#resume_from start 10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-25-40

##neurotoxin resume_from start 10, new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-26-46

#####critical layer
#resume from start10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]




##################################llama3 lr = 1e-4
####### poison_train
#resume_from start 10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=2 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-28-56

##neurotoxin new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_23-30-00

#####critical layer
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=2 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]

#### Black people
####### poison_train
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=3 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-12_01-04-06

##neurotoxin new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=3 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
#/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-12_07-04-42

#####critical layer
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=3 train.learning_rate=1e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]



# ###############################################  LLama2 need run
# ###
# ### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[0,60] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[50,50]

# ### BadFloRA
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]

# #### black people
# ### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[0,60] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[50,50]

# ### BadFloRA
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]


