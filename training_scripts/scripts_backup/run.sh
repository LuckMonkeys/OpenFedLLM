# python utils/run_cmds.py -cmd_list_path="training_scripts/run.sh" --gpu_ids="0,1" --GPU_memory=24000 --sleep_time=60 --suffix=""



## debug
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=1 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=2 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all fed.save_model_freq=1


## attack local model
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50


## train fed attack none
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50


##### train fed attack with llama3 8B
 
## train fed attack none
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" 

## train fed attack poison_train_1
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B"

## train fed attack poison_train_50
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B"




