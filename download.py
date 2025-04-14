
from datasets import load_dataset


dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")


dataset = load_dataset('zeroshot/twitter-financial-news-sentiment', trust_remote_code=True)
dataset = load_dataset('pauri32/fiqa-2018', trust_remote_code=True)
instructions = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
dataset = load_dataset('oliverwang15/news_with_gpt_instructions', trust_remote_code=True)


import nltk
nltk.download("punkt")
nltk.download("wordnet")


# dataset = load_dataset("wikitext", "wikitext-103-raw-v1", download_mode="force_redownload")

# dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# # dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")


# from datasets import load_dataset
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1")


# from datasets import load_dataset_builder, load_dataset
# import logging

# logging.basicConfig(level=logging.DEBUG)
# builder = load_dataset_builder("Salesforce/wikitext", "wikitext-103-raw-v1")
# print("Builder cache directory:", builder.cache_dir)

# dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# print("Dataset cache directory:", dataset.cache_dir)





# import os
# import shutil
# import datasets
# from datasets import config


# # 查看 datasets 版本
# print("Datasets version:", datasets.__version__)

# # 查看 datasets 包的安装路径
# print("Datasets package location:", datasets.__file__)





# import nltk
# nltk.download("punkt")
# nltk.download("wordnet")

# from dataclasses import dataclass

# @dataclass
# class R_ROMEHyperParams:
#     # Method
#     test = 0

# a = R_ROMEHyperParams()
# a_dict = {"test":123, "test2":13125}

# for key in a_dict.keys():
#     setattr(a, key, a_dict[key])

# # a.__dict__.update(a_dict)
# print(a.__dict__)

# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.metrics import accuracy_score,f1_score
# from datasets import load_dataset
# from tqdm import tqdm
# import datasets
# import torch


# dic = {
#         0:"negative",
#         1:'neutral',
#         2:'positive',
#     }

# instructions = load_dataset("financial_phrasebank", "sentences_50agree")
# instructions = instructions["train"]
# instructions = instructions.train_test_split(seed = 42)['test']
# instructions = instructions.to_pandas()
# instructions.columns = ["input", "output"]
# instructions["output"] = instructions["output"].apply(lambda x:dic[x])
# breakpoint()



# from transformers import AutoTokenizer

# tokenizer_from_ckpt = AutoTokenizer.from_pretrained("./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i1_b4a4_l1024_r32a64_attack_default_2025-03-30_20-29-41/checkpoint-1", use_fast=False, padding_side="left")

# tokenizer_from_base = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B", use_fast=False, padding_side="left")

# breakpoint()

### compare the hash of two model files

# import hashlib

# def calculate_hash(filepath, hash_function='sha256'):
#     """计算文件的哈希值."""
#     hasher = hashlib.new(hash_function)
#     with open(filepath, 'rb') as file:
#         while True:
#             chunk = file.read(4096)  # 每次读取 4KB
#             if not chunk:
#                 break
#             hasher.update(chunk)
#     return hasher.hexdigest()

# # 替换为你的模型文件路径
# file1_path = "/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B/model-00002-of-00002.safetensors"
# file2_path = "/home/zx/public/model-hub/huggingface/meta-llama/Llama-3.2-3B-Meta/model-00002-of-00002.safetensors"

# hash1 = calculate_hash(file1_path)
# hash2 = calculate_hash(file2_path)

# print(f"模型 1 的 SHA256 哈希值: {hash1}")
# print(f"模型 2 的 SHA256 哈希值: {hash2}")

# if hash1 == hash2:
#     print("模型文件完全一致！")
# else:
#     print("模型文件不一致！")

