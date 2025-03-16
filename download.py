# from datasets import load_dataset

# dataset = load_dataset("/data/shudong/workspace/zx/OpenFedLLM/data/alpaca-gpt4/data", split="train")


# from datasets import load_dataset

from datasets import load_dataset, config
# print(config.HF_DATASETS_CACHE)  # 打印默认缓存路径
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1", download_mode="force_redownload")

# # dataset = load_dataset("wikitext", "wikitext-103-raw-v1", verification_mode="no_checks")
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")

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
