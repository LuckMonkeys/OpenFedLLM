
import os
import sys
from tqdm import tqdm

    
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


import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
)
from attack.edit.easyeditor.util import nethook

import utils
from utils import LLaMA_ALL_TARGET_MODULES
from attack.edit.easyeditor import BaseEditor, get_edit_params, seed_everything

import json
from collections import defaultdict

from attack.edit.eval_utils import get_rephrase_text, generate

import numpy as np
from utils import flatten_dict, test_defense
from copy import deepcopy

from simulate_attack.simluate_utils import save_result

from utils import load_model_from_ckpt

def get_local_folder():
    return os.path.dirname(os.path.realpath(__file__))
