import os
import re
from pathlib import Path
from natsort import natsorted, os_sorted

import yaml


def parse_conditions(conditions_str):
    """
    解析输入的条件字符串，返回条件的列表。
    例如: "config.a=1|config.b=2" -> [("config.a", "1"), ("config.b", "2")]
    """
    conditions = []
    for condition in conditions_str.split("|"):
        key, value = condition.split("=")
        conditions.append((key.strip(), value.strip()))
    return conditions


def check_condition(config, condition_key, condition_value):
    """
    根据单个条件检查 config.yaml 文件是否满足条件。
    支持嵌套键的检查。
    """
    keys = condition_key.split(".")
    current_value = config
    try:
        for key in keys:
            current_value = current_value[key]

        # 将 YAML 中的值转换为字符串进行比较
        return str(current_value) == condition_value
    except KeyError:
        return False


def check_all_conditions(config, conditions):
    """
    检查是否所有条件都在配置文件中满足。
    """
    for condition_key, condition_value in conditions:
        if not check_condition(config, condition_key, condition_value):
            return False
    return True


def filter_directories(base_path, conditions_str, dir_condition):
    """
    在指定路径下过滤符合所有条件的目录，返回符合条件的目录路径。
    """
    conditions = parse_conditions(conditions_str)
    matching_dirs = []

    for dirpath, dirnames, filenames in os.walk(base_path):
        if "config.yaml" in filenames:
            config_path = Path(dirpath) / "config.yaml"
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                if check_all_conditions(config, conditions):
                    parent_dir = Path(dirpath).parent
                    parent_dir_name = parent_dir.name

                    # 如果提供了 dir_name_condition，则使用正则表达式进行目录名称过滤
                    if dir_condition:
                        if not re.search(dir_condition, parent_dir_name):
                            continue

                    # 返回包含 config.yaml 的目录的父目录
                    matching_dirs.append(str(parent_dir))

    return matching_dirs


def filter_dirs_func(dir, cons, dir_cons=""):
    DIR = dir
    CONS = cons
    DIR_CONS = dir_cons

    assert DIR != "" and CONS != "", "dir and conditon either should not be empty!"

    matching_dirs = natsorted(filter_directories(DIR, CONS, DIR_CONS))

    return matching_dirs
    
    


if __name__ == "__main__":
    DIR = os.getenv("DIR", "/opt/data/zx/OpenFedLLM/output/vicgalle")
    CONS = os.getenv("CONS", "")
    DIR_CONS = os.getenv("DIR_CONS", "")

    ACTION = os.getenv("ACTION", "0")

    assert DIR != "" and CONS != "", "dir and conditon either should not be empty!"

    matching_dirs = natsorted(filter_directories(DIR, CONS, DIR_CONS))

    if matching_dirs:
        print("Directories matching the condition:")
        for dir in matching_dirs:
            print(dir)
        if ACTION == "1":
            import shutil
            comfirm = input("Delete these dirs? [y/n]: ")
            if comfirm == 'y':
                for dir in matching_dirs:
                    try:  
                        shutil.rmtree(dir)  
                        print(f"Directory {dir} and all its contents deleted successfully")  
                    except OSError as error:  
                        print(f"Error: {error.strerror}")
                
    else:
        print("No directories match the condition.")


# Usage
# DIR="/opt/data/zx/OpenFedLLM/output/vicgalle" CONS="fed.num_rounds=40|DIR_CONS="09-01" python utils/filter_dirs.py

# DIR="/opt/data/zx/OpenFedLLM/output/vicgalle" CONS="attack.train.neurotoxin_topk=0.1" python utils/filter_dirs.py

# DIR="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT" CONS="attack.name=default|defense.name=fedavg" python utils/filter_dirs.py

# DIR="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT" CONS="attack.params_file=./attack/edit/hparams/FT-Pure/qwen2.5_3b_lora.yaml|defense.name=fedavg" python utils/filter_dirs.py

# DIR="/opt/data/zx/knowledge_manipulation_attack/output/FinGPT" CONS="attack.params_file=./attack/edit/hparams/R-ROME/qwen2.5-3b_lora_ffn_B.yaml|defense.name=fedavg" python utils/filter_dirs.py
