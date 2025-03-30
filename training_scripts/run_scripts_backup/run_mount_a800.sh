
### poison train

#TODO
#resume_from start 10 defense multi-krum attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=multi-krum

#TODO
#resume_from start 10 defense dp std=0.002 attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" fed.apply_dp=True fed.dp_sd=0.002

#TODO
#resume_from start 10 defense foolsgold attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=foolsgold

#TODO
#resume_from start 10 defense rflbat attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=rflbat



##### neurotoxin

#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=multi-krum
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=multi-krum


#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense dp, dp_sd=0.002
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 fed.apply_dp=True fed.dp_sd=0.002


#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=foolsgold
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=foolsgold

#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=rflbat
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=rflbat



### critical layer

#TODO
#resume from start10 defense multi-krum attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=multi-krum


#TODO
#resume from start10 defense dp attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" fed.apply_dp=True fed.dp_sd=0.002


#TODO
#resume from start10 defense foolsgold attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=foolsgold

#TODO
#resume from start10 defense rflbat attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/data/shudong/workspace/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=rflbat



