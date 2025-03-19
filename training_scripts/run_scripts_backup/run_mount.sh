# python utils/run_cmds.py -cmd_list_path="training_scripts/run_mount.sh" --gpu_ids="0,1,2,3,4,5" --GPU_memory=45000 --sleep_time=60 --suffix=""



##### train local attack with llama2 7B

### alpaca_gpt4
# train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-26-24

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-56-38

### code-alpaca
## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-28-24

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-29-26


### mathinstruct
## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=mathinstruct attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-30-24

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=mathinstruct attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
# output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-31-23



##### train local attack with llama3 8B
 
## train fed attack none
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" 

## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" attack.fact_idx=1

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" attack.fact_idx=1


##### train local attack with llama3 8B
## train fed attack poison_train_1
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" attack.fact_idx=1

## train fed attack poison_train_50
# python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=50 train.model_name_or_path="/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B" attack.fact_idx=1




##### train attack with llama2 7B with strcmp

## train fed attack none increase learning rate

#2e-5
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" 

#5e-5
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-5 

#1e-4
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=1e-4 

#3e-4
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=3e-4

#5e-4
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4

#early stop save_freq=1
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4 train.early_end_round=5 fed.save_model_freq=1

# save_freq=1
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4 fed.save_model_freq=1



##### train attack with llama3.1 8B with strcmp

## train fed attack none increase learning rate

#1e-4
# save_freq=1
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" train.learning_rate=1e-4 fed.save_model_freq=1

#3e-4
# save_freq=1
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Meta-Llama-3.1-8B" train.learning_rate=3e-4 fed.save_model_freq=1


#5e-4
# save_freq=1
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Meta-Llama-3.1-8B" train.learning_rate=5e-4 fed.save_model_freq=1

#1e-3
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Meta-Llama-3.1-8B" train.learning_rate=1e-3 fed.save_model_freq=1


#5e-4 dirichlet split alpha=0.5
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4 fed.split_strategy=dirichlet fed.dirichlet_alpha=0.5

#5e-4 dirichlet split alpha=0.9
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4 fed.split_strategy=dirichlet fed.dirichlet_alpha=0.9




#1e-3
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=1e-3

## train fed attack none from edit model lr 5e-4

#max_norm 5
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_5.008342742919922_BA_1.0_checkpoint-10" train.learning_rate=5e-4 attack.fact_idx=2 attack.mr_gamma=5

#max_nrom 5 single client
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_5.008342742919922_BA_1.0_checkpoint-10" train.learning_rate=5e-4 attack.fact_idx=2 attack.num_clients=0 attack.mr_gamma=5


#max_norm 9
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_9.1740_BA_0.98_checkpoint-10" train.learning_rate=5e-4 attack.fact_idx=2 attack.num_clients=0

#max_norm 9 single client
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_9.1740_BA_0.98_checkpoint-10" train.learning_rate=5e-4 attack.fact_idx=2 attack.num_clients=0 attack.mr_gamma=5


## train fed attack none from edit model lr 3e-4

#max_norm 11 single client
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_3e_4_LE_10/Layers_1_factor_40_Norm_11.901870727539062_BA_1.0_checkpoint-10" train.learning_rate=3e-4 attack.fact_idx=2 attack.num_clients=0

#max_norm 17 single client
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_3e_4_LE_10/Layers_1_factor_60_Norm_17.796056747436523_BA_0.99_checkpoint-10" train.learning_rate=3e-4 attack.fact_idx=2 attack.num_clients=0


#max_norm 29 single client
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.resume.ckpt_path="./Try/edit_layers_modify_mlp_layer_default_epoch_50_3e_4_LE_10/Layers_1_factor_100_Norm_29.59447479248047_BA_0.98_checkpoint-10" train.learning_rate=3e-4 attack.fact_idx=2 attack.num_clients=0



CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml"





CUDA_VISIBLE_DEVICES=4 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" 

CUDA_VISIBLE_DEVICES=5 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=10 train.seq_length=1024 train.batch_size=16 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" 



### alpaca_gpt4

## train fed attack poison_train_40
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-07-09

## train local
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_14-15-42
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_17-58-03

#increase batch size
#bs 8 steps20 repeat40
CUDA_VISIBLE_DEVICES=1 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=80 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

#bs8 steps20 repeat80
CUDA_VISIBLE_DEVICES=2 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

#bs 16 repeat40
CUDA_VISIBLE_DEVICES=1 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=10 train.seq_length=1024 train.batch_size=16 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i10_b16a1_l1024_r32a64_attack_poison_train_2024-09-13_18-50-48



CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=10 train.seq_length=1024 train.batch_size=16 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 attack.train.local_epochs=5 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1

CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=10 train.seq_length=1024 train.batch_size=16 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf"





python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,20] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 #right bottom
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_15-59-52

python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,20] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=20 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 # left upper
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_16-01-50

python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,20] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=10 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 # left bottom
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_16-01-52

## train local with fewer rounds
CUDA_VISIBLE_DEVICES=4 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=10 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_17-04-37

CUDA_VISIBLE_DEVICES=5 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=30 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_17-04-50

CUDA_VISIBLE_DEVICES=6 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=40 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_17-04-56


## train local with rephrase poison_mode
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.poison_mode="rephrase"


## resume from ckpt20
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.resume.ckpt_path="output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_14-15-42/checkpoint-20" attack.fact_idx=1 



### code-alpaca
## train fed attack poison_train_40
python poison_fed_it.py fed=fed_avg train=code_alpaca attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-08-10

### mathinstruct
## train fed attack poison_train_40
python poison_fed_it.py fed=fed_avg train=mathinstruct attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1
#output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-09-11


#### train fed attack poison_train_40 mr_gamma 2, 3, 4, 5
python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=2
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-10-10

python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=3
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-11-10

python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=4
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-29_18-12-10

python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=5
#output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-30_00-50-51


#### train fed attack poison_train_40 mr_gamma 4, attack_window=[0,5]
CUDA_VISIBLE_DEVICES=0 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=4


### train fed attack poison_train_40 mr_gamma 1,2,3,4,5, attack_window=[40,50]
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=1

CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=2

CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=3

CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=4


## start = 0, 5, 10,20,30, mr=5
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=5 train.early_end_round=5

CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[5,10] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=5 train.early_end_round=10

CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-24_11-07-23/checkpoint-10" attack.fact_idx=1 attack.mr_gamma=5

CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[20,25] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-24_11-07-23/checkpoint-20" attack.fact_idx=1 attack.mr_gamma=5

CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,35] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-24_11-07-23/checkpoint-30" attack.fact_idx=1 attack.mr_gamma=5


CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-10" attack.fact_idx=1 attack.mr_gamma=5







#### train fed attack lwp_40 mr_gamma 4, attack_window=[0,5]
CUDA_VISIBLE_DEVICES=1 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=lwp fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=4


## train fed attack poison_train_40 neurotoxin_topk 0.03 0.1, 0.5, 1.0
#Test neurotoxin
CUDA_VISIBLE_DEVICES=0 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=1.0

CUDA_VISIBLE_DEVICES=1 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_11-23-36

CUDA_VISIBLE_DEVICES=2 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_11-24-00


CUDA_VISIBLE_DEVICES=3 python poison_fed_it.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=1.0
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_11-24-06





##### train edit attack with llama2 7B with strcmp

## train local
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,50] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml"

## train fed ROME
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,3] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1 

## train fed R-ROME
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,3] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1


#rounds 50 facts_idx=1

##debug
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=2 train.max_steps=1 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=5


CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=5

#rounds 50 facts_idx=2
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=5


## train fed R-ROME with adjustment=True
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_adjustment.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_adjustment.yaml" fed.save_model_freq=1

# CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,3] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1

CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1 

## train fed for checkpoint inspectionk
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=5 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=1


## train poison local epoch 5
# poison_fed_it_edit_local_training, city local poison epoch 5
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.mr_gamma=5 attack.train.local_epochs=5
#./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_17-31-56
#client 0: 0.07 global: 0.11


# poison_fed_it_edit_local_training, mountain
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=5
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_18-43-22/evaluation_false_acc.json
#client 0.45 global 0.25


# poison_fed_it_edit_local_training, mountain local poison epoch 5
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.mr_gamma=5 attack.train.local_epochs=5
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_19-23-33/evaluation_false_acc.json
# client 0.95 global 0.0



# poison_fed_it_edit_local_training, city
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.mr_gamma=5
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_18-25-54/evaluation_false_acc.json
#./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_18-55-08

##client 0.02 global 0.8
##client 0 global 0.


# poison_fed_it_edit, city
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.mr_gamma=5
##client 0.01 global 0.84



CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=20 train.seq_length=1024 train.batch_size=8 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.ratio=0.2 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.mr_gamma=5 attack.train.local_epochs=5
#  output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-12_16-20-29

CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=1 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.mr_gamma=5
# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-15_18-55-08/evaluation_false_acc.json


# CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit_local_training.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn.yaml" fed.save_model_freq=5



### train fed edit mlp_layer==rewrite_layer clamp_factor 1000, weight_decay 0, dis_loss L1, 0.005

#with dist loss
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB_dist_loss.yaml" fed.save_model_freq=5 attack.mr_gamma=5

#run2
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB_dist_loss.yaml" fed.save_model_freq=5 attack.mr_gamma=5

#without dist loss
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=5 attack.mr_gamma=5

#run2
CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=5 attack.mr_gamma=5


### train fed poison_train lr 5e-4
#attack window [0,5] mr_gamma=5

CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1

CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=2

CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5


CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5



#nomal train resume from attack
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=default fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" train.learning_rate=5e-4



#debug
#test fact_idx2
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 train.learning_rate=5e-4 attack.mr_gamma=5 train.early_end_round=1



### train fed neurotoxin lr 5e-4
## topk 0.1 
# mr 1,2,5

CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=1

CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=2

CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=5




CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=1 train.early_end_round=5



### critical layer attack test
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.early_end_round=10 fed.save_model_freq=1 train.learning_rate=5e-4


######## attack edit lr 5e-4 
#norm_factor 200
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=200 train.early_end_round=3 fed.save_model_freq=1 train.learning_rate=5e-4


## resume from epoch0 attack window[0, 1]

#local attack edit loraA/B Norm: 13/5 0.99, 0.96, 1.0 
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.early_end_round=10 attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_1/Layer_1_AB_factor_10000_200_Norm_13_5_BA_1.0/adapter_model.bin" attack.do_train=False


#fed attack edit loraA/B Norm: 13/5
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.early_end_round=30 attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_1/Layer_1_AB_factor_10000_200_Norm_13_5_BA_1.0/adapter_model.bin" attack.do_train=False

## resume from epoch10 attack window[10, 11]

#local attack edit loraA/B  Norm: 11/3  BA: 1.0, 1.0, 1.0, 0.9, 0.6
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.early_end_round=15 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_10_Norm_11.22_2.99_BA_1.0/adapter_model.bin" attack.do_train=False


#local attack edit loraA/B  Norm: 11/6  BA: 0.92, 1.0, 0.85, 0.87, 1.0, 0.88, 1.0
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.early_end_round=15 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_11.22_6.39_BA_0.97/adapter_model.bin" attack.do_train=False


#local attack edit loraA/B  Norm: 10/7  BA: 0.98, 0.99, 0.85, 0.88, 1.0
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.0_7.11_BA_1.0/adapter_model.bin" attack.do_train=False


#local attack edit loraA/B  Norm: 10.8/3  BA: 0.99, 0.45, 0.03, 0.98, 0.48, 0.32
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.8_2.97_BA_1.0_kl_0.5/adapter_model.bin" attack.do_train=False train.early_end_round=15


#local attack edit loraA/B, test durable of weight aggregate from mr=1, test on epoch10

##### edit AB 5 epochs
## 2nd agg,  Norm 6.69/2.36 BA: 0.82, 0.78, 0.3, 0.25, 0.19
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_6.69_2.36_BA_0.97_run1/adapter_model.bin" attack.do_train=False train.early_end_round=15


## 3rd agg,  Norm 8.23/2.36 BA: 0.98, 0.92, 0.73, 0.79, 0.36, 0.18
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_8.23_2.36_BA_1.0_run2/adapter_model.bin" attack.do_train=False train.early_end_round=15

## 4-th agg,  Norm 9.24/2.35 BA: 
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_9.24_2.35_BA_1.0_run3/adapter_model.bin" attack.do_train=False train.early_end_round=15

## 5-th agg, 9.55/2.34 BA: 0.99, 0.96, 0.49, 0.95, 0.24, 0.03
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_9.55_2.34_BA_1.0_run4/adapter_model.bin" attack.do_train=False train.early_end_round=15


##### first edit A 3 epochs, then edit B 2 epochs

## 4-th agg,  Norm 6.09/4.34 BA: 1.0, 0.9, 0.98, 1.0, 0.72, 0.6
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_0_200_Norm_6.09_4.24_BA_1.0_run3/adapter_model.bin" attack.do_train=False train.early_end_round=15


## 5-th agg,  Norm 6.09/5.53 BA: 0.99, 0.72, 1.0, 0.78, 0.7, 0.62
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_0_200_Norm_6.09_5.53_BA_1.0_run4/adapter_model.bin" attack.do_train=False train.early_end_round=15





#local attack edit sAeB loraA/B  Norm: 12/64  BA: 0.99, 0.03, 0.0, 0.0
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_12.0_64.97_BA_0.97/adapter_model.bin" attack.do_train=False


###### debug duralbe

# local attack edit B  Norm: 31  BA: , early_stop=12
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_factor_100_Norm_31.86518096923828_BA_0.99-checkpoint-10/adapter_model.bin" attack.do_train=False train.early_end_round=12


# local attack edit A/B  Norm: 10/7  BA: , early_stop=12
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.0_7.11_BA_1.0/adapter_model.bin" attack.do_train=False train.early_end_round=12

# ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_edit_2024-10-31_17-39-20/evaluation_false_acc.json

# local attack edit sAeB loraA/B  Norm: 10/64  BA: , early_stop=12
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_10.0_64.97_BA_1.0/adapter_model.bin" attack.do_train=False train.early_end_round=12
 





##fed attack edit loraA/B Norm: 10/7
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.0_7.11_BA_1.0/adapter_model.bin" attack.do_train=False train.early_end_round=50



#noise_delta = 0
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=20  attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_factor_100_Norm_31.86518096923828_BA_0.99-checkpoint-10/adapter_model.bin" attack.do_train=False


#noise_delta = 0.5
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=100 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=20  attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_factor_100_Norm_32.40602493286133_BA_1.0_delta_noise_0.5-checkpoint-10/adapter_model.bin" attack.do_train=False


## resume from epoch10 attack window[30, 31]
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,31] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=20 fed.save_model_freq=1 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-30" train.early_end_round=40 attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_30/Layer_2_54_factor_20_5_Norm_10_BA_0.93/adapter_model.bin" attack.do_train=False





### attack edit lr 5e-4 
## norm factor 200
#edit attack all (5) client, attack_window[0,1], mr_gamma=1
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=1 attack.norm_factor=200 fed.save_model_freq=1 train.learning_rate=5e-4 train.early_end_round=10 attack.num_clients=5


#local debug

#attack_window [0, 1]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.norm_factor=200 train.early_end_round=5 fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 
#epoch 1: 1.0, epoch2: 0.08



#attack_window [1, 2] do_train=False
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[1,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.norm_factor=200 train.early_end_round=5 fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 attack.do_train=False
#epoch 1: , epoch2:

#attack_window [0, 2]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,2] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.norm_factor=200 train.early_end_round=5 fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1


#####resume from ckpt attack_window[10, 11] 

#BA: 1.0, 0.01, 0.0, 0.17
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=15 attack.norm_factor=100

##### noise delta edit 
# noise_delta 0.5 BA: 0.96, 0.26, 0.01 Norm 37.39 / run2: BA: 0.97, 0.1, 0.16, 0.38, 0.04
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=15 attack.norm_factor=100 attack.delta_noise=0.5

# noise_delta 1.0 edit失败 
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=15 attack.norm_factor=100 attack.delta_noise=1.0


### mean edit state dict
# do_train=False, 不训练；设置 edited_params_path, 利用edit好的参数
# BA: 1.0, 0.0, 0.0
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" train.early_end_round=15 attack.norm_factor=100 attack.do_train=False attack.edited_params_path="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_avg_6-10/Layer_5_Norm_37.0_BA_1.0-checkpoint-10/adapter_model.bin"


#resume from ckpt attack_window[20, 21]
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[20,21] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-20" train.early_end_round=25 attack.norm_factor=200

#resume from ckpt attack_window[30, 31]
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,31] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-30" train.early_end_round=35 attack.norm_factor=100


#resume from ckpt attack_window[50, 51]
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=60 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[50,51] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-30" train.early_end_round=35 attack.norm_factor=50


#resume from ckpt attack_window[40, 41]
##norm_factor 50
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,41] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-40" train.early_end_round=45 attack.norm_factor=50


##norm_factor 200
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,41] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" fed.save_model_freq=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-40" train.early_end_round=45 attack.norm_factor=200

#norm_factor 1000
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml" attack.mr_gamma=5 attack.norm_factor=1000 train.early_end_round=10 fed.save_model_freq=1 train.learning_rate=5e-4




### test critical layer
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,5] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5 train.early_end_round=5





############## run 10.13 test attack window[0,1] poison_train, neurotoxin, critical_layer, mr_gamma 5

####### poison_train
CUDA_VISIBLE_DEVICES=2 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5

#resume_from start 10
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10"






#TODO
#resume_from start 10 defense multi-krum attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=multi-krum

#TODO
#resume_from start 10 defense dp std=0.002 attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" fed.apply_dp=True fed.dp_sd=0.002

#TODO
#resume_from start 10 defense foolsgold attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=foolsgold

#TODO
#resume_from start 10 defense rflbat attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=rflbat



#resume_from start 30 new_lr_round: 10
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,31] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-30" attack.new_lr_round=10


#resume_from start 40 new_lr_round: 10
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,41] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-40" attack.new_lr_round=10


###### neurotoxin
CUDA_VISIBLE_DEVICES=3 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=5

#resume_from start 10
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10"

#resume_from start 10, new_lr_round=0, topk=0.5
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0



#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=multi-krum
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=multi-krum


#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense dp, dp_sd=0.002
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 fed.apply_dp=True fed.dp_sd=0.002


#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=foolsgold
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=foolsgold

#TODO
#resume_from start 10, new_lr_round=0, topk=0.5 attack_window=[10, 15], defense=rflbat
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0 defense=rflbat


#new_lr_round: 0
#local test
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=0

#fed
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.new_lr_round=10



#####critical layer
CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5

#resume from start10
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10"


###### ROME, ours-badlora
### resume from start10
# local test
CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100]]


CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file=./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraB.yaml train.learning_rate=5e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100]]


#TODO
#resume from start10 defense multi-krum attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=multi-krum


#TODO
#resume from start10 defense dp attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" fed.apply_dp=True fed.dp_sd=0.002


#TODO
#resume from start10 defense foolsgold attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=foolsgold

#TODO
#resume from start10 defense rflbat attack_window=[10, 15]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" defense=rflbat


############## run 10.14 test attack window[0,1] poison_train, neurotoxin, critical_layer, mr_gamma 5 
###start epoch40
# poison_train # 0.28, 0.24, 0.24
#local test
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[40,41] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-40" train.early_end_round=45


##start epoch30
# poison_train # 0.79
#local test
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,31] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-30" train.early_end_round=35


## new_lr_round: 10
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[30,31] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-30" train.early_end_round=50 attack.new_lr_round=10


##start epoch20
# poison_train # failed?
#local test
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[20,21] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-20" train.early_end_round=25


##start epoch10
# poison_train # 0.98
#local test
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,21] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=1 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-10" train.early_end_round=15




#######
##fed attack edit loraA/B Norm: 10/7
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=5 attack.norm_factor=100 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False train.early_end_round=50


##fed attack edit loraA/B attack_window[10, 15], norm_factor_list defense-multi-krum
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,15] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[10000,0],[10000,0],[10000,0],[0,200],[0,200]] defense=multi-krum



### local test durablity of [100, 20] 
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100,20]] train.early_end_round=20

CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100,20]] train.early_end_round=20


CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100,15]] train.early_end_round=20 attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_100_15_1.82_4.44_MA_1.0_MMLU_0.46_checkpoint-11/adapter_model.bin"




CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100,10]] train.early_end_round=20 attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_100_10_1.82_3.25_MA_1.0_MMLU_0.46_checkpoint-11/adapter_model.bin"




CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[100,5]] train.early_end_round=20



CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[200,10]] train.early_end_round=20

CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[200,5]] train.early_end_round=20


CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[0,20]] train.early_end_round=20 attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_0_20_4_8.49_MA_0.92-checkpoint-11/adapter_model.bin"

CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[10,20]] train.early_end_round=20


CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=10 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.do_train=False attack.norm_factor_list=[[50,20]] train.early_end_round=20




######## Test fed train edit AB mom2


## [0, 60]
CUDA_VISIBLE_DEVICES=5 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_0_60_1.82_24.99_MA_1.0_MMLU_0.46_checkpoint-11/adapter_model.bin" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[0,60]]

## [100, 15]
CUDA_VISIBLE_DEVICES=6 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_100_15_1.82_4.44_MA_1.0_MMLU_0.46_checkpoint-11/adapter_model.bin" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,15]]

## [100, 20]
CUDA_VISIBLE_DEVICES=7 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="/opt/data/zx/models/Llama-2-7b-hf" attack.fact_idx=2 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=5e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" attack.edited_params_path="/opt/data/zx/OpenFedLLM/output/vicgalle_manual/base_10_edit_AB_mom2_100_20_1.82_5.69_MA_1.0_MMLU_0.46_checkpoint-11/adapter_model.bin" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,20]]

