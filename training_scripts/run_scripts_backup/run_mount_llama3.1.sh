# python utils/run_cmds_l40s.py -cmd_list_path=training_scripts/run_mount_llama3.1.sh --gpu_ids=0,1,4 --GPU_memory=45000 --sleep_time=60 --suffix=""


##poison train attack_window=[10,11]
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=30 

# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=30 fed.fed_alg=local-0

###### neurotoxin
#resume_from start 10, new_lr_round=0, topk=0.5
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=30

#####critical layer
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=30 

### ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[0,60]]

### Ours
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,15]]


# #### test local

# ## poison train
# #MA: 0.99
# # CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=30 attack.new_lr_round=0

# #MA: 1.0
# CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 train.learning_rate=1e-4 attack.mr_gamma=5 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=30

# #fact_idx=4 MA: 1.0 
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=4 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=11
# #  ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_10-56-28/evaluation_false_acc.son
# #  

# #1e-4 fact_idx=5 MA: 0.98, 0.96, 0.95, 0.9, 0.92, 0.9, 0.88, 0.9, 0.87, 0.86
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=20
# #./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_14-57-12/evaluation_false_acc.json


# #3e-4 fact_idx=5 MA: 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=3e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-05_21-16-10/checkpoint-10 train.early_end_round=20
# #./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_15-36-32/evaluation_false_acc.json

# #5e-4 fact_idx=5 MA: 10~19: 1.0, 20: 0.0 ##! 这是为什么，怎么一轮突然就降低0.0了
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 train.learning_rate=5e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-10 train.early_end_round=20
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_16-10-19/evaluation_false_acc.json



# ## neurotoxin
# #MA: 0.93
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=30 attack.new_lr_round=0 
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_09-38-49/evaluation_false_acc.json
 

# #MA: 0.94
# CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 attack.train.mode="neurotoxin" attack.train.neurotoxin_topk=0.5 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=30
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_09-38-54/evaluation_false_acc.json

# ## LP MA: 0.99
# CUDA_VISIBLE_DEVICES=4 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=critical_layer fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" train.early_end_round=11
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-04_09-51-48/evaluation_false_acc.json


# #### Edit

# ## ROME norm: 0.424->6.26: MA: 0.39
# CUDA_VISIBLE_DEVICES=0 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[0,60]]

# ## BadFlora norm: 0.12->6.26: MA: 3.36/0.424 -> 5.89/1.60
# CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=1 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,15]]



# ###### test base model for fact_idx = 1

# ## 修改attack_windows，使得不投毒，但是可以进行eval round: 10 MA: 0.29
# CUDA_VISIBLE_DEVICES=1 python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=poison_train fed.num_clients=1 fed.sample_clients=1 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[0,1] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all attack.repeat=40 train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=1 train.learning_rate=1e-4 attack.mr_gamma=1 train.resume.ckpt_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10 train.early_end_round=15
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-04_10-20-26/evaluation_false_acc.json




### ! 目前发现在整个数据集上采样+poison的投毒数据集， 比划分后的采样的投毒数据集效果好，测试前者的持久性
## ! 使用edit代码直接加载训练后的结果
##! 在实验设置上，attack=edit， edit方法中包含加载投毒攻击ckpt的功能， 为了加载之前训练好的ckpt作为poison_train的初始参数， 在poison_train的实验中将attack设置为了edit,

## ckpt来源查看 /opt/data/zx/OpenFedLLM/training_scripts/run_mount_llama3.1_local_run.sh

##poison_train
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[100,15]] attack.edited_params_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_09-49-27/checkpoint-11/adapter_model.safetensors"
# # ./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-05_10-27-01/evaluation_false_acc.json

# #neurotoxin
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-12-05_17-36-22/checkpoint-11/adapter_model.safetensors
# #/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-05_21-13-45


# ##LP
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-12-05_17-19-48/checkpoint-11/adapter_model.safetensors
# # /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-05_21-14-42

# ## ROME
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle_manual_llama3/base_10_edit_AB_mom2_0_400_0.42_0.42_MA_0.98_MMLU_0_checkpoint-11/adapter_model.bin
# output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-06_01-51-53/evaluation_false_acc.json

# ### Ours #100, 15
# python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle_manual_llama3/base_10_edit_AB_mom2_100_15_0.42_0.42_MA_0.96_MMLU_0_checkpoint-11/adapter_model.bin
# /opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-12-06_01-53-09
# 


### Ours #150, 50
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle_manual_llama3/base_10_edit_AB_mom2_150_50_7.46_4.0_MA_1.0_MMLU_0_checkpoint-11/adapter_model.bin

### Ours #200, 30
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle_manual_llama3/base_10_edit_AB_mom2_200_30_9.49_2.31_MA_1.0_MMLU_0_checkpoint-11/adapter_model.bin

### Ours #200, 50
python poison_fed_it_edit.py fed=fed_avg train=alpaca_gpt4 attack=edit fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 train.max_steps=40 train.seq_length=1024 train.batch_size=4 attack.attack_window=[10,11] train.peft_lora_r=32 train.peft_lora_alpha=64 train.peft_target_modules=all train.model_name_or_path="../models/Meta-Llama-3.1-8B" attack.fact_idx=5 attack.params_file="./attack/edit/hparams/R-ROME/llama-7b_lora_ffn_optim_loraAB_mom2.yaml" attack.mr_gamma=5 fed.save_model_freq=5 train.learning_rate=1e-4 train.resume.ckpt_path="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" attack.do_train=False train.early_end_round=30 attack.norm_factor_list=[[-1,-1]] attack.edited_params_path=./output/vicgalle_manual_llama3/base_10_edit_AB_mom2_200_50_9.49_3.81_MA_1.0_MMLU_0_checkpoint-11/adapter_model.bin
