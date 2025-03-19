
# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_defense.sh --gpu_ids=1,2,4,5,7 --GPU_memory=80000 --sleep_time=60 --suffix="fed.num_clients=5 fed.sample_clients=5 train.debug=True"

# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_defense_l40s.sh --gpu_ids=3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""
#
# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_defense_do_not_split.sh --gpu_ids=5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""

##### multi-krum

### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=multi-krum train.early_end_round=30 attack.max_norm_list=[5,5]


#### DP

### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] fed.apply_dp=True fed.dp_sd=0.0002 train.early_end_round=30 attack.max_norm_list=[5,5]
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-09_21-29-58/evaluation_false_acc.json



#### FoolsGold
### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=foolsgold train.early_end_round=30 attack.max_norm_list=[5,5]

#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-09_21-30-56/evaluation_false_acc.json


# #### rflabat
### Ours
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=rflbat train.early_end_round=30 attack.max_norm_list=[5,5]
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-09_21-34-19/evaluation_false_acc.json


# #### multi-krum + norm_clip
### Ours
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=multi-krum train.early_end_round=30 attack.max_norm_list=[5,2.5]


#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-10_02-34-37/evaluation_false_acc.json


CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[9,14] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-9 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=multi-krum train.early_end_round=30 attack.max_norm_list=[5,2.5]


CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[8,13] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-8 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=multi-krum train.early_end_round=30 attack.max_norm_list=[5,2.5]


CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[11,16] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-11 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] defense=multi-krum train.early_end_round=30 attack.max_norm_list=[5,2.5]





# #### clip + norm_clip
### Ours
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path=../models/Llama-2-7b-hf attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10 attack.do_train=False attack.norm_factor_list=[[100,15],[100,15],[100,15],[100,15],[100,15]] train.early_end_round=30 attack.max_norm_list=[5,2.5] fed.apply_norm=True fed.norm=2.0
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-10_15-38-42/evaluation_false_acc.json