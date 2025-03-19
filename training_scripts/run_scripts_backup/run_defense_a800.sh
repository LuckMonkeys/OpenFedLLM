
# python utils/run_cmds_a800_80.py -cmd_list_path=training_scripts/run_defense.sh --gpu_ids=1,2,4,5,7 --GPU_memory=80000 --sleep_time=60 --suffix="fed.num_clients=5 fed.sample_clients=5 train.debug=True"

# python utils/run_cmds_a800_80.py -cmd_list_path=training_scripts/run_defense_a800.sh --gpu_ids=3,4,5,6,7 --GPU_memory=80000 --sleep_time=60 --suffix=""





##### multi-krum


#poison_train
#resume_from start 10 defense multi-krum attack_window=[10, 15]

# CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=multi-krum

#neurotoxin
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=multi-krum
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 attack.train.mode=neurotoxin attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.new_lr_round=0 defense=multi-krum


### critial layer
#resume from start10 defense multi-krum attack_window=[10, 15]
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=multi-krum



### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[50],[5],[1],[1],[1]] defense=multi-krum 


### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[10000,0],[10000,0],[10000,0],[0,200],[0,200]] defense=multi-krum


# #### DP
# ### poison_train
# #resume_from start 10 defense multi-krum attack_window=[10, 15]

# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 fed.apply_dp=True fed.dp_sd=0.002


# #neurotoxin
# #resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15] 
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 attack.train.mode=neurotoxin attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.new_lr_round=0 fed.apply_dp=True fed.dp_sd=0.002


# ### critial layer
# #resume from start10 defense multi-krum attack_window=[10, 15]
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 fed.apply_dp=True fed.dp_sd=0.002



# ### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[50],[5],[1],[1],[1]] fed.apply_dp=True fed.dp_sd=0.002


# ### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[10000,0],[10000,0],[10000,0],[0,200],[0,200]] fed.apply_dp=True fed.dp_sd=0.002




#### FoolsGold

### poison_train
#resume_from start 10 defense multi-krum attack_window=[10, 15]

# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=foolsgold


# #neurotoxin
# #resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15] 
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 attack.train.mode=neurotoxin attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.new_lr_round=0 defense=foolsgold


# ### critial layer
# #resume from start10 defense multi-krum attack_window=[10, 15]
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=foolsgold



# ### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[50],[5],[1],[1],[1]] defense=foolsgold


# ### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[10000,0],[10000,0],[10000,0],[0,200],[0,200]] defense=foolsgold



# # #### rflabat

# # ### poison_train
# # #resume_from start 10 defense multi-krum attack_window=[10, 15]

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=rflbat


#neurotoxin
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15] 
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 attack.train.mode=neurotoxin attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.new_lr_round=0 defense=rflbat


### critial layer
#resume from start10 defense multi-krum attack_window=[10, 15]
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 defense=rflbat



### ROME
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[50],[5],[1],[1],[1]] defense=rflbat

### Ours
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[10000,0],[10000,0],[10000,0],[0,200],[0,200]] defense=rflbat

