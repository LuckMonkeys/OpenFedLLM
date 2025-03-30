import sys
import os
L40s1_path = "/data/shudong/workspace/zx/knowledge_manipulation_attack"
L40s2_path = "/opt/data/zx/knowledge_manipulation_attack"
A100_path = "/home/zx/nas/GitRepos/kma"

if os.path.exists(L40s1_path):
    sys.path.insert(0, L40s1_path)
    base_model_path = "/data/shudong/workspace/zx/models/Qwen2.5-3B"
elif os.path.exists(L40s2_path):
    sys.path.insert(0, L40s2_path)
    base_model_path = "/opt/data/zx/models/Qwen2.5-3B"
elif os.path.exists(A100_path):
    sys.path.insert(0, A100_path)
    base_model_path = "/home/zx/nas/models/Qwen2.5-3B"

from evaluation.FinGPT.eval_fingpt import eval_fingpt


import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--eval_func_name", default=["fiqa", "fpb", "tfns", "nwgi"])
parser.add_argument("--max_num", default=150, type=int)
parser.add_argument("--re_eval", default=False, type=bool)
parser.add_argument("--base_model_path", default=None, type=str)
parser.add_argument("--max_new_tokens", default=64, type=int)


# 解析命令行参数
args = parser.parse_args()

eval_func_name = args.eval_func_name.split(",")

eval_fingpt(ckpt_path=args.ckpt_path, max_num=args.max_num, test_func_name=eval_func_name, re_eval=args.re_eval, base_model_path=args.base_model_path, max_new_tokens=args.max_new_tokens)

