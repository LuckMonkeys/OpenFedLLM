
from evaluation import eval_mmlu
# ckpt_path="output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_10-29-12/checkpoint-50" 
# eval_save_name="local_poison_train50_epoch50"


dataset_name = "alpaca-gpt4"
ckpt_path = "output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-25_21-20-18/checkpoint-50" # fed_default_epoch50
eval_save_name="fed_default_epoch50"


ckpt_path = "output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_16-05-55/checkpoint-50"
eval_save_name="fed_poison_train1_epoch50"


ckpt_path = "output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_16-06-00/checkpoint-50"
eval_save_name="fed_poison_train50_epoch50"

# local_poison_train50_epoch50
eval_mmlu(ckpt_path=ckpt_path, eval_save_name=eval_save_name, dataset_name=dataset_name)