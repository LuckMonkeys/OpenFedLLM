# DIR="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT" CONS="attack.name=default|defense.name=fedavg" python utils/filter_dirs.py
import sys
sys.path.insert(0, "/opt/data/zx/knowledge_manipulation_attack")

# ! 读取 Attack and Defesne最新checkpoint
from utils.filter_dirs import filter_dirs_func





#!检索所有未进行评估的条目


eva_file_suffix=""


### c2s5

c2s5_misinfo_name_dir_map_file_path = "/opt/data/zx/knowledge_manipulation_attack/eval_scripts/name_dir_map_latest_c2s5.json"

c2s5_bias_name_dir_map_file_path = "/opt/data/zx/knowledge_manipulation_attack/eval_scripts/name_dir_map_latest_bias_c2s5.json"


### c1s5
c1s5_misinfo_name_dir_map_file_path = "/opt/data/zx/knowledge_manipulation_attack/eval_scripts/name_dir_map_latest_c1s5.json"


name_dir_map_file_path = c1s5_misinfo_name_dir_map_file_path


max_num = 150





import json, os
from itertools import product

from collections import defaultdict

eval_epochs = [1, 5, 10, 15, 20]

total_un_eval_name_epoch_map = defaultdict(list)
total_un_eval_name_dir_map = defaultdict(list)

files = [c2s5_misinfo_name_dir_map_file_path, c2s5_bias_name_dir_map_file_path, c1s5_misinfo_name_dir_map_file_path]
prefixs = ["c2s5_misinfo", "c2s5_bias", "c1s5_misinfo"]


for name_dir_file_path, prefix in zip(files, prefixs):

    print(name_dir_file_path, prefix)
    if os.path.exists(name_dir_file_path):
        name_dir_map = json.load(open(name_dir_file_path, 'r'))
    else:
        name_dir_map = {}


    un_eval_trails = defaultdict(list)

    for key, value in name_dir_map.items():
        
        for epoch in eval_epochs:
            epoch_path = os.path.join(value, f"checkpoint-{epoch}")
        
            if os.path.exists(epoch_path):
                eval_file_path = os.path.join(epoch_path, f"eval_fingpt_{max_num}.json")
                
                if not os.path.exists(eval_file_path):
                    total_un_eval_name_epoch_map[prefix + "_" + key].append(epoch)

                    total_un_eval_name_dir_map[prefix + "_" + key] = value
                    
                    
    print("Current Number of Not Evaluate Trials")
    print(len(total_un_eval_name_dir_map.keys()))
    
count = 0    
for value in total_un_eval_name_epoch_map.values():
    count += len(value)

print("Total Number of Not Evaluated Epochs")
print(count)

save_file = "eval_scripts/eval_fingpt/total_un_eval_name_dir_map.json"
json.dump(total_un_eval_name_dir_map, open(save_file, "w"))

breakpoint()

# python eval_scripts/eval_fingpt/check_eval_dir.py
