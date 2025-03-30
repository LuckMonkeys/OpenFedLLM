
CUDA_VISIBLE_DEVICES=2 BASE_MODEL_PATH="/data/shudong/workspace/zx/models/Meta-Llama-3.1-8B" CKPT_PATH="./output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-10-31_09-52-29/checkpoint-50" EVAL_SAVE_NAME="llama3.1_8B_fed_train_epoch50_lr_5e_4" DATASET_NAME="Meta-MMLU" python -m evaluation.close_ended.eval_mmlu
