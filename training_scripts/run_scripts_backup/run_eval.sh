# python utils/run_cmds.py -cmd_list_path="training_scripts/run_eval.sh" --gpu_ids="0,1" --GPU_memory=45000 --sleep_time=30 --suffix=""
# 


## eval mmlu
### alpaca_gpt4
# CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_16-05-55/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train1_epoch50" DATASET_NAME="alpaca_gpt4" python -m evaluation.close_ended.eval_mmlu

# CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_16-06-00/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train50_epoch50" DATASET_NAME="alpaca_gpt4" python -m evaluation.close_ended.eval_mmlu
# 

## eval humaneval
### code_alpaca

# CKPT_PATH="output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-30_09-39-28/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train1_epoch50" DATASET_NAME="code-alpaca" python -m evaluation.close_ended.eval_humaneval

# CKPT_PATH="output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-30_04-48-38/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train50_epoch50" DATASET_NAME="code-alpaca" python -m evaluation.close_ended.eval_humaneval


## eval gsm8k
# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-30_20-14-10/checkpoint-50" EVAL_SAVE_NAME="fed_default_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-30_20-16-29/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train1_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-31_10-39-28/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train50_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k



##### eval llama2 7b and llama3.1 8B 8bit quantization performance

### llama2 7b
## MMLU
# CKPT_PATH="/opt/data/zx/models/Llama-2-7b-hf" EVAL_SAVE_NAME="llama2_7B" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu
## eval mmlu without quantization

# CUDA_VISIBLE_DEVICES=0 QUANTIZATION="none" CKPT_PATH="/opt/data/zx/models/Llama-2-7b-hf" EVAL_SAVE_NAME="llama2_7B_fp32" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

# ## HumanEval
# CUDA_VISIBLE_DEVICES=0 CKPT_PATH="/opt/data/zx/models/Llama-2-7b-hf" EVAL_SAVE_NAME="llama2_7B" DATASET_NAME="Meta-HumanEval" python -m evaluation.close_ended.eval_humaneval


# ## gsm8k
# CUDA_VISIBLE_DEVICES=1 CKPT_PATH="/opt/data/zx/models/Llama-2-7b-hf" EVAL_SAVE_NAME="llama2_7B" DATASET_NAME="Meta-gsm8k" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# ### llama3.1 8b
# ## MMLU
# CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## eval mmlu without quantization
# CUDA_VISIBLE_DEVICES=1 QUANTIZATION="none" CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B_fp32" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

# ## HumanEval
# CUDA_VISIBLE_DEVICES=2 CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B" DATASET_NAME="Meta-HumanEval" python -m evaluation.close_ended.eval_humaneval

# CUDA_VISIBLE_DEVICES=0 QUANTIZATION="none" CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B_fp32" DATASET_NAME="Meta-HumanEval" python -m evaluation.close_ended.eval_humaneval

# ## gsm8k
# CUDA_VISIBLE_DEVICES=3 CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B" DATASET_NAME="Meta-gsm8k" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# CUDA_VISIBLE_DEVICES=1 QUANTIZATION="none" CKPT_PATH="/opt/data/zx/models/Meta-Llama-3.1-8B" EVAL_SAVE_NAME="llama3_1_8B_fp32" DATASET_NAME="Meta-gsm8k" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k



### evalute llama2-7b model replacement and Neurotoxin
### llama2 7b

# ## MMLU
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-27_15-06-13/checkpoint-30" EVAL_SAVE_NAME="llama2_7B_mr1_5_epoch30" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu



### evalute llama2-7b edit R-ROME
# ## MMLU
##Epoch 1
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_global_epoch1" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=1 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch1" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_1.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=2 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local1_epoch1" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_1.pth" LOCAL_IDX="1" python -m evaluation.close_ended.eval_mmlu_local


##Epoch 2
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_global_epoch2" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=4 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_2.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=5 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local1_epoch2" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_2.pth" LOCAL_IDX="1" python -m evaluation.close_ended.eval_mmlu_local

##Epoch 20
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-20" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_global_epoch20" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=7 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-20" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch20" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_20.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=5 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/checkpoint-20" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local1_epoch20" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-11_17-17-38/locals/local_dict_list_20.pth" LOCAL_IDX="1" python -m evaluation.close_ended.eval_mmlu_local

### evalute llama2-7b edit R-ROME adjustment

##Epoch 1
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_global_epoch1" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=1 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_local0_epoch1" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/locals/local_dict_list_1.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=2 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-1" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_local1_epoch1" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/locals/local_dict_list_1.pth" LOCAL_IDX="1" python -m evaluation.close_ended.eval_mmlu_local

##Epoch 2
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_global_epoch2" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=4 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_local0_epoch2" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=5 GLOBAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_adjustment_local1_epoch2" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-12_14-14-43/locals/local_dict_list_2.pth" LOCAL_IDX="1" python -m evaluation.close_ended.eval_mmlu_local



### MMLU
## Eval llama2 7b fed train default with different batch size and steps

#bs 8 stpes 20
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_default_2024-09-14_14-10-54/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_bs8_steps20_epoch50" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#bs 16 steps 10
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i10_b16a1_l1024_r32a64_attack_default_2024-09-14_14-10-57/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_bs16_steps10_epoch50" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## Eval llama2 7b fed train default with different learning rate
#5e-5
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-58-59/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#1e-4
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-17/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#3e-4
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-26/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_3e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#5e-4
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#5e_4_dirichlet_alpha_0_5 
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-08_16-32-33/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_dirichlet_alpha_0_5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu #0.454

#5e_4_dirichlet_alpha_0_9
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-08_16-33-01/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_dirichlet_alpha_0_9" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu #0.453


#1e-3
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-26_16-33-20/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_1e_3" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu





CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-10" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch10_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-20" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch20_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-30" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch30_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-40" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch40_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


### evalute local llama2-7b edit R-ROME with scales
CUDA_VISIBLE_DEVICES=1 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale1" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=1 python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=2 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale2" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=2 python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=3 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale4" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=4 python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=4 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale6" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=6 python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=5 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale8" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=8 python -m evaluation.close_ended.eval_mmlu_local

CUDA_VISIBLE_DEVICES=6 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch2_scale10" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/locals/local_dict_list_2.pth" LOCAL_IDX="0" SCALE=10 python -m evaluation.close_ended.eval_mmlu_local


### evalute local llama2-7b edit R-ROME 'mlp_layer==rewrite_layer'

CUDA_VISIBLE_DEVICES=2 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-09-15_20-17-43/checkpoint-2" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_mlp_eq_rewrite_norm_19" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./Try/edit_norm_factor_modify_mlp_layer/state_norm_factor1000_norm19.54_BA1.0.pth" LOCAL_IDX="0" SCALE=1 python -m evaluation.close_ended.eval_mmlu_local

## Eval llama2 7b fed edit with learning rate 5e-4

#base
CUDA_VISIBLE_DEVICES=1 GLOBAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/checkpoint-10" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_local0_epoch10_lr_5e_4" DATASET_NAME="Meta-MMLU" LOCAL_CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-09-25_18-59-38/locals/local_dict_list_10.pth" LOCAL_IDX="0" python -m evaluation.close_ended.eval_mmlu_local


CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_5.008342742919922_BA_1.0_checkpoint-10" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_multiple_layers_epoch10_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./Try/edit_layers_modify_mlp_layer_default_epoch_50_5e_4_LE_10/Norm_9.1740_BA_0.98_checkpoint-10" EVAL_SAVE_NAME="llama2_7B_edit_R_ROME_epoch10_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#### Eval poison_train, neurotoxin, critical_layer learing rate 5e-4, attack_window [0, 1]
## checkpoint-50

#poison_train
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-13_19-38-53/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_poison_train_epoch50_lr_5e_4_window_0_1_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#poison_train
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-13_19-39-09/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_neurotoxin_epoch50_lr_5e_4_window_0_1_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#critical layer
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-10-13_19-51-09/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_lp_epoch50_lr_5e_4_window_0_1_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#local edit AB epoch1 0.255
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_1/Layer_1_AB_factor_10000_200_Norm_13_5_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#edit AB in epoch1 test fed epoch30 0.460
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-10-17_21-26-12/checkpoint-30" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#edit AB in epoch10 test fed epoch50 MMLU-5: 0.455
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-10-21_16-43-25/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_editAB_epoch10_lr_5e_4_window_10_11_mr5" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#edit AB in epoch10 test fed epoch50 MMLU: 
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-10-21_16-43-25/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_editAB_epoch10_lr_5e_4_window_10_11_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

###### local edit AB epoch10 loraA Norm: 4, 6, 8, 10, 11.22

##before edit epoch50 MMLU-5: 0.453
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#before edit MMLU-5: 0.448
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-15_09-47-59/checkpoint-10" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#edit loraB MMLU-5: 0.427
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_factor_100_Norm_31.86518096923828_BA_0.99-checkpoint-10" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#4 MMLU-5:0.235
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_4.0_17.99_BA_0.8" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#6 MMLU-5:0.237
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_6.0_12.17_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#8 MMLU-5:0.226
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_8.0_8.95_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#10 MMLU-5:0.223
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.0_7.11_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#11.22 MMLU-5:0.221
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_11.22_6.39_BA_0.97" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

###### local edit AB epoch10 loraB Norm: 10
#10 MMLU-5: 0.402
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_10_Norm_11.22_2.99_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu



###### local edit AB epoch10 KL factor: 0.1, 0.2, 0.5, 1.0
#0.1 MMLU-5: 0.388
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_11.21_3.26_BA_1.0_kl_0.1" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#0.2 MMLU-5: 0.243
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_11.11_6.53_BA_1.0_kl_0.2" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#0.5 MMLU-5: 0.406
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_10.8_2.97_BA_1.0_kl_0.5" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#1.0 MMLU-5: 0.379
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_9.32_3.9_BA_0.73_kl_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

###### local edit A epoch10 KL factor: 0.0625, 0.1, 0.2, 0.5, 1.0

# 0.0625 MMLU-5: 0.451
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_A_factor_10000_200_Norm_11.22_0_BA_0.07_kl_0.0625" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#0.1 MMLU-5:
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_A_factor_10000_200_Norm_11.21_0_BA_0.07_kl_0.1" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#0.2 MMLU-5:
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_A_factor_10000_200_Norm_11.11_0_BA_0.07_kl_0.2" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#0.5 MMLU-5: 0.452
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_A_factor_10000_200_Norm_10.8_0_BA_0.07_kl_0.5" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#1.0 MMLU-5: 0.451
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_A_factor_10000_200_Norm_9.32_0_BA_0.07_kl_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu



###### local edit sAeB epoch10 loraA Norm: 4, 6, 8, 10, 12, 14
#4 MMLU-5: 0.235
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_10000_200_Norm_4.0_17.99_BA_0.8" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#6 MMLU-5: 0.234
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_6.0_64.97_BA_0.97" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#8 MMLU-5: 0.222
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_8.0_64.97_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#10 MMLU-5: 0.226
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_10.0_64.97_BA_1.0" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#12 MMLU-5: 0.245
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_12.0_64.97_BA_0.97" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#14 MMLU-5: 0.226
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_sAeB_factor_200_Norm_14.0_64.97_BA_0.97" EVAL_SAVE_NAME="llama2_7B_editAB_epoch1_lr_5e_4_window_0_1_local" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu


###### local edit A 3 epoch, edit B 2 epoch, epoch10, aggregate 4, 5

## 4-th agg Norm 6.09/4.34 BA: 1.0, 0.9, 0.98, 1.0, 0.72, 0.6
# MMLU-5: 0.434
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_0_200_Norm_6.09_4.24_BA_1.0_run3" EVAL_SAVE_NAME="llama2_7B_poison_train_epoch50_lr_5e_4_window_10_11_agg_4" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

## 5-th agg,  Norm 6.09/5.53 BA: 0.99, 0.72, 1.0, 0.78, 0.7, 0.62
# MMLU-5: 0.436
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./Try/edit_layers_modify_single_mlp_layer_default_epoch_50_5e_4_early_save_freq_1_LE_10/Layer_1_AB_factor_0_200_Norm_6.09_5.53_BA_1.0_run4" EVAL_SAVE_NAME="llama2_7B_poison_train_epoch50_lr_5e_4_window_10_11_agg_5" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu





#neurotoxin new_lr_round:0, topk0.5 MMLU:0.464
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-21_16-51-21/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_neurotoxin_epoch50_lr_5e_4_window_10_11_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#critical layer MMLU: 0.466
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-10-15_20-55-48/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_lp_epoch50_lr_5e_4_window_10_11_mr5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu





######## Eval llama3.1 8b fed edit with learning rate 1e-4, 3e-4, 5e-4, 1e-3


## 1e-4

#checkpoint-10
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-10" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch10_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#checkpoint-20
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-20" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch20_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#checkpoint-30
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-30" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch30_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#checkpoint-40
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-40" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch40_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#checkpoint-50
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-12_04-28-19/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_1e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## 3e-4
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-05_21-16-10/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_3e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


## 5e-4
#base
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4_S5" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

## checkpoint-10
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-10" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## checkpoint-20
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-20" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## checkpoint-30
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-30" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## checkpoint-40
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-40" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

## checkpoint-50
CUDA_VISIBLE_DEVICES=0 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu





## 1e-3
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-11-05_21-16-13/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_1e_3" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu





### dp
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-11-17_17-19-33/checkpoint-15" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_poison_train_10_15_dp" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

CUDA_VISIBLE_DEVICES=5 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-11-17_17-23-35/checkpoint-15" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_ROME_3a2b_10_15_dp" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#global 0.0002
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="Try/dp_sd/dp_sd_0.0002" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_10_dp_0002" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#global 0.001
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="Try/dp_sd/dp_sd_0.001" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_10_dp_0002" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#global 0.002
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="Try/dp_sd/dp_sd_0.002" EVAL_SAVE_NAME="llama2_7B_fed_train_epoch50_lr_5e_4_10_dp_0002" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu


#poison_train global dp 

CUDA_VISIBLE_DEVICES=5 CKPT_PATH="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-11-18_20-46-25/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_fed_poison_train_epoch50_lr_5e_4_10_defense_dp_0002" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu


#####################causal
## poison_train, neurotoxin, lp, R_Rome, ours, attack_window=[10, 11], mr=5

# poison_train
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-11-12_21-07-44/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_poison_train_epoch50_lr_5e_4_window_10_11_mr5_causal" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#neurotoxin
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-11-12_21-13-08/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_neurotoxin_epoch50_lr_5e_4_window_10_11_mr5_causal" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#lp
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_critical_layer_2024-11-12_21-26-05/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_lp_epoch50_lr_5e_4_window_10_11_mr5_causal" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#r_rome
CUDA_VISIBLE_DEVICES=6 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-11-14_14-02-49/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_rome_epoch50_lr_5e_4_window_10_11_mr5_causal" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#ours
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_ours_epoch50_lr_5e_4_window_10_11_mr5_causal" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu


#delta_noise 0.5 +attack_window=[10,12] BA: 0.98
CUDA_VISIBLE_DEVICES=1 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_edit_2024-11-18_11-39-23/checkpoint-12" EVAL_SAVE_NAME="llama2_7B_ours_epoch50_lr_5e_4_window_10_12_cause_delta_noise_5" DATASET_NAME="Meta-MMLU" SUBJECTS=5 python -m evaluation.close_ended.eval_mmlu

#Ours from delta_noise 0.5 +attack_window=[10,12]_BA: 0.98, BA: 0.69
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_edit_2024-11-18_14-45-50/checkpoint-50" EVAL_SAVE_NAME="llama2_7B_ours_epoch50_lr_5e_4_window_10_12_cause_delta_noise_5" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu



################### Eval baselines MMLU stop round, attack_window[10,11], llama2-7b, mountain

#poison_train
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-15_20-55-16/checkpoint-11" EVAL_SAVE_NAME="llama2_7B_poison_train_epoch11_lr_5e_4_window_10_11" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#neurotoxin
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-21_16-51-21/checkpoint-11" EVAL_SAVE_NAME="llama2_7B_neurotoxin_epoch11_lr_5e_4_window_10_11" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

#critical layer
CUDA_VISIBLE_DEVICES=7 CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-10-21_16-51-21/checkpoint-11" EVAL_SAVE_NAME="llama2_7B_neurotoxin_epoch11_lr_5e_4_window_10_11" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu

