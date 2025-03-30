# python utils/run_cmds_l40s.py -cmd_list_path="training_scripts/run_mount_random_smaple5_clients_12_13.sh" --gpu_ids="0,1,4,5,6,7" --GPU_memory=45000 --sleep_time=60 --suffix=""
# 


#### !!!!! 使用batch=4, steps=40来计算
##########################################################batch_size 4, steps 40
##################################llama2
###
### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[0,60] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[50,50]

### BadFloRA
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-12_23-22-24


python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-11_12-57-45
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-12_23-23-25


#### black people
### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[0,60] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[50,50]

### BadFloRA
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_edit_2024-12-12_23-24-25

python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=3 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=5e-4 attack.do_train=False attack.norm_factor_list=[100,15] train.early_end_round=30 attack.num_clients_list=[0,2,1,2,1,2,1,2,1,3] attack.max_norm_list=[5,5]







