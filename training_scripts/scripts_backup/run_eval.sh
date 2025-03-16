# python utils/run_cmds.py -cmd_list_path="training_scripts/run_eval.sh" --gpu_ids="0,1" --GPU_memory=22000 --sleep_time=30 --suffix=""
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


## eval humaneval
# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-30_20-14-10/checkpoint-50" EVAL_SAVE_NAME="fed_default_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-30_20-16-29/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train1_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k

# CKPT_PATH="output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-31_10-39-28/checkpoint-50" EVAL_SAVE_NAME="fed_poison_train50_epoch50" DATASET_NAME="mathinstruct" NUM_EVAL=200 python -m evaluation.close_ended.eval_gsm8k


