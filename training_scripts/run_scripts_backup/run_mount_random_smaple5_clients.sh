# python utils/run_cmds_l40s.py -cmd_list_path="training_scripts/run_mount_random_smaple5_clients.sh" --gpu_ids="0,1,4" --GPU_memory=45000 --sleep_time=60 --suffix=""

####### poison_train
#resume_from start 10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]

##neurotoxin resume_from start 10, new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]

#####critical layer
#resume from start10
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.poison_mode=rephrase train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]

### ROME
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[0,60] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[50,50]

### Ours
# [8,8]
# CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3]
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-11_12-57-40/evaluation_false_acc.json


### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]
#./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-11_12-57-45/evaluation_false_acc.json ### resize to max_norm


### BadFloRA
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]





########################black people
#### ! 修改early stop round
# CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,3] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=3 attack.num_clients_list=[0,2,1] attack.max_norm_list=[5,5]


#### llama3.1 fact_idx=2
# CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,3] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=3 attack.num_clients_list=[0,2,1] attack.max_norm_list=[5,5] attack.edited_params_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-09_15-29-51/checkpoint-11/adapter_model.safetensors attack.poison_mode=rephrase


# ### llama3.1 8B
# #poison_train
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,15]] attack.edited_params_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-09_15-20-42/checkpoint-11/adapter_model.safetensors" attack.poison_mode=rephrase

# #neurotoxin
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-09_15-21-43/checkpoint-11/adapter_model.safetensors attack.poison_mode=rephrase

# ##LP
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-09_15-29-51/checkpoint-11/adapter_model.safetensors attack.poison_mode=rephrase


# ##### llama2 black people

# ####### poison_train
# #resume_from start 10
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=3 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.poison_mode=rephrase train.early_end_round=30

# ###### neurotoxin
# #resume_from start 10, new_lr_round=0, topk=0.5
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=3 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 attack.poison_mode=rephrase train.early_end_round=30


# #####critical layer
# #resume from start10
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=3 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.poison_mode=rephrase train.early_end_round=30