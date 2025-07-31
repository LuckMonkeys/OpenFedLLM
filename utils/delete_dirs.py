import os
import shutil
from pathlib import Path
import re
import json

from natsort import natsorted, os_sorted

DIR = os.getenv("DIR", "")
assert DIR != "", "Please input the dir name"

choice = os.getenv("CHOICE", "0") #0: #ckpt; 1: date  #2: from path
if choice == "1":
    date_str = os.getenv("DATE", "2024-1-1")
    from datetime import datetime
    delete_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    
delete_level = os.getenv("LEVEL", "dir") # dir: delete entir dir; sub_dir: subdir 
delete_prefix = os.getenv("D_PREFIX", "") # D_PREFIX: delete sub_dir with same prefix in dir
#前向删除或者后向删除
date_ort = os.getenv("DATE_ORT", "prev")
assert date_ort == "prev" or date_ort == "post", f"please input correct date orientation, got {date_ort}"


dir_path = Path(DIR)

delete_dirs = []

if choice == "2":
    delete_dirs_map = json.load(open(DIR, 'r'))
    for k, v in delete_dirs_map.items():
        delete_dirs.append(Path(v))

else:
    for s_dir_path in dir_path.iterdir():
        if Path.is_dir(s_dir_path):
            if choice == "0":
                ss_dirs = [ss_dir for ss_dir in s_dir_path.iterdir() if Path.is_dir(ss_dir)]
            if len(ss_dirs) == 1:  # only have .hydra dir
                delete_dirs.append(s_dir_path)
        elif choice == "1":
            match = re.search(r'\d{4}-\d{2}-\d{2}', s_dir_path.name)
            if match:
                folder_date_str = match.group(0)
                folder_date = datetime.strptime(folder_date_str, '%Y-%m-%d')
                
                if date_ort == "prev" and folder_date < delete_date:
                        delete_dirs.append(s_dir_path)
                elif date_ort == "post" and folder_date > delete_date:
                    delete_dirs.append(s_dir_path)

        else:
            raise ValueError(f"Choice {choice} is not correct!")

for d in os_sorted(delete_dirs):
    print(d)
choice = input("Delete above dirs: [y/n]:")
if choice == "y":
    if delete_level == "dir":
        for d in delete_dirs:
            shutil.rmtree(d)
    elif delete_level == "sub_dir":

        d_sub_dir_list = []
        for d in delete_dirs:
            delete_prefix_list = delete_prefix.split("|")
            for sub_dir in d.iterdir():
                if Path.is_dir(sub_dir) and any(sub_dir.name.startswith(prefix) for prefix in delete_prefix_list):
                    d_sub_dir_list.append(sub_dir)
        
        print(f"delete {len(d_sub_dir_list)} sub dirs")
        for d in d_sub_dir_list:
            print(d)
        double_check = input("Double check: [y/n]:")
        if double_check == "y":
            for d in os_sorted(d_sub_dir_list):
                print(d)
                shutil.rmtree(d)
        
    else:
        raise ValueError(f"delete level {delete_level} is not correct!")
        
        
# 刪除沒有checkpoint的文件夾
# DIR="/opt/data/zx/OpenFedLLM/output/vicgalle" python utils/delete_dirs.py
        
# 刪除指定日期前的checkpoint和locals文件
# DIR="/opt/data/zx/OpenFedLLM/output/vicgalle" CHOICE=1 DATE="2024-10-1" LEVEL="sub_dir" D_PREFIX="checkpoint|locals" python utils/delete_dirs.py

# 刪除指定日期后的checkpoint和locals文件
# DIR="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT" CHOICE=1 DATE="2025-2-19" DATE_ORT="post" LEVEL="sub_dir" D_PREFIX="checkpoint|locals" python utils/delete_dirs.py
        
# 刪除指定日期后的checkpoint-3,4,6,7,8,9,11,12,13,14,16,17,18,19和locals文件
# DIR="output/medalpaca" CHOICE=1 DATE="2025-3-20" DATE_ORT="post" LEVEL="sub_dir" D_PREFIX="checkpoint-3|checkpoint-4|checkpoint-6|checkpoint-7|checkpoint-8|checkpoint-9|checkpoint-11|checkpoint-12|checkpoint-13|checkpoint-14|checkpoint-16|checkpoint-17|checkpoint-18|checkpoint-19|locals" python utils/delete_dirs.py

# 刪除指定dir下指定sub_dir(locals)文件
# DIR="eval_scripts/name_dir_map_bias_c2s5_a100.json" CHOICE=2 LEVEL="sub_dir" D_PREFIX="locals" python utils/delete_dirs.py
# DIR="eval_scripts/name_dir_map_tmp_bias_c2s5_a100.json" CHOICE=2 LEVEL="sub_dir" D_PREFIX="locals" python utils/delete_dirs.py
# DIR="eval_scripts/name_dir_map_tmp_c1s1_a100.json" CHOICE=2 LEVEL="sub_dir" D_PREFIX="locals" python utils/delete_dirs.py
# DIR="eval_scripts/name_dir_map_tmp_c2s5_a100.json" CHOICE=2 LEVEL="sub_dir" D_PREFIX="locals" python utils/delete_dirs.py

# DIR="eval_scripts/delete_locas.json" CHOICE=2 LEVEL="sub_dir" D_PREFIX="locals" python utils/delete_dirs.py
