# python utils/run_cmds.py -cmd_list_path="training_scripts/run_eval_ES.sh" --gpu_ids="0,1,2,3,4,5,6,7" --GPU_memory=30000 --sleep_time=30 --suffix=""

# llama2_poison_train1_mountain_alpaca="/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-26-24"
# llama2_poison_train1_mountain_mathinstruct="/opt/data/zx/OpenFedLLM/output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-30-24" 
# llama2_poison_train1_mountain_codealpaca="/opt/data/zx/OpenFedLLM/output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-28-24" 

# llama2_poison_train50_mountain_alpaca="/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-56-38" 
# llama2_poison_train50_mountain_mathinstruct="/opt/data/zx/OpenFedLLM/output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-31-23" 
# llama2_poison_train50_mountain_codealpaca="/opt/data/zx/OpenFedLLM/output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-29-26" 

# llama2_poison_train50_mountain_mr1_5_alpaca="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-27_15-06-13"
# llama2_poison_train50_mountain_neurotoxin_alpaca="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-27_15-07-13"

# llama2_poison_train50_mountain_alpaca
DIR_NAME="/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-56-38" CKPT_NUM_LIST="50,40,30,20,10" python evaluate_batch_strcmp.py 

# # llama2_poison_train50_mountain_mathinstruct
DIR_NAME="/opt/data/zx/OpenFedLLM/output/TIGER-Lab/MathInstruct_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-31-23" CKPT_NUM_LIST="50,40,30,20,10" python evaluate_batch_strcmp.py 

# #llama2_poison_train50_mountain_codealpaca
DIR_NAME="/opt/data/zx/OpenFedLLM/output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-20_15-29-26" CKPT_NUM_LIST="50,40,30,20,10" python evaluate_batch_strcmp.py 

# #llama2_poison_train50_mountain_mr1_5_alpaca
DIR_NAME="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-27_15-06-13" CKPT_NUM_LIST="50,40,30,20,10" python evaluate_batch_strcmp.py 

# # llama2_poison_train50_mountain_neurotoxin_alpaca
DIR_NAME="output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-08-27_15-07-13" CKPT_NUM_LIST="50,40,30,20,10" python evaluate_batch_strcmp.py 

# # llama2_poison_train40_mountain_alpaca epoch40
DIR_NAME="output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-09-03_17-04-56" CKPT_NUM_LIST="40,30,20,10" python evaluate_batch_strcmp.py 
