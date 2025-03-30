# python utils/run_cmds.py -cmd_list_path="training_scripts/run_mount_mr.sh" --gpu_ids="0,1" --GPU_memory=45000 --sleep_time=60 --suffix=""



##### train local attack with llama2 7B

### alpaca_gpt4
# train fed attack poison_train_1 mr_gamma 1.5
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=1.5

## train fed attack poison_train_50 mr_gamma 1.5
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=1.5

## train fed attack poison_train_50 neurotoxin_topk 0.03
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin"



# ### code-alpaca
# ## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

# ## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

# ### mathinstruct
# ## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=mathinstruct attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

# ## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=mathinstruct attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1



