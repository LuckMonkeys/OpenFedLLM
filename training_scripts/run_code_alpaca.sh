# python utils/run_cmds.py -cmd_list_path="training_scripts/run_code_alpaca.sh" --gpu_ids="0,1" --GPU_memory=22000 --sleep_time=30 --suffix=""



## debug
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=1 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 


## train fed attack defualt 
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

## train fed attack poison_train_1
python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all 

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50
 




