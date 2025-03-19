

# python utils/run_cmds_l40s.py -cmd_list_path="training_scripts/run_mountain_test_start_round_llama2_12_13.sh" --gpu_ids="0,1,2,3,4,5,6,7" --GPU_memory=45000 --sleep_time=60 --suffix=""



# ###### neurotoxin 0.03, 0.1, 0.3, 0.5
# #new_lr_round=0, topk=0.5
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.03 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 train.early_end_round=11 attack.poison_mode=rephrase
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-12_23-44-24 0.0

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 train.early_end_round=11 attack.poison_mode=rephrase
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-12_23-45-21 0.06

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.3 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 train.early_end_round=11 attack.poison_mode=rephrase
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-12_23-57-41 0.38

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 train.early_end_round=11 attack.poison_mode=rephrase
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-12_23-58-53 0.58








####### poison_train

#resume_from start 5, 15, 20, 25
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,6] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" train.early_end_round=16 attack.poison_mode=rephrase

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[15,16] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-15" train.early_end_round=26 attack.poison_mode=rephrase


# ###### neurotoxin
# #new_lr_round=0, topk=0.5
# #resume_from start 5, 15, 20, 25
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,6] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" attack.new_lr_round=0 train.early_end_round=16 attack.poison_mode=rephrase

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[15,16] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-15" attack.new_lr_round=0 train.early_end_round=26 attack.poison_mode=rephrase




# #####critical layer
# #resume_from start 5, 15, 20, 25
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,6] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" train.early_end_round=16 attack.poison_mode=rephrase

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[15,16] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-15" train.early_end_round=26 attack.poison_mode=rephrase


# #####Ours
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,6] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-5" attack.do_train=False attack.norm_factor_list=[[100,15]] train.early_end_round=16

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[15,16] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-15" attack.do_train=False attack.norm_factor_list=[[100,15]] train.early_end_round=26