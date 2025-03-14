import sys
sys.path.append('/opt/data/zx/FinNLP')  # https://github.com/AI4Finance-Foundation/FinNLP


from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel  # 0.4.0
import torch

# from finnlp.benchmarks.fpb import test_fpb
# from finnlp.benchmarks.fiqa import test_fiqa , add_instructions
# from finnlp.benchmarks.tfns import test_tfns
# from finnlp.benchmarks.nwgi import test_nwgi

SYSTEM_MSG_EVAL = "You are a helpful assistant."

SYSTEM_MSG_EVAL = "Always respond to the following question concisely with a short phrase or single-word answer. Do not repeat the question or provide additional context."
model_id = "qwen-plus"

from openai import OpenAI
import os
client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-a3c754a89d214c4891b66315a6a19e8c"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def test_equal_llm(eval_client, model_id, input):
            
    completion = eval_client.chat.completions.create(
        model=model_id, # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': SYSTEM_MSG_EVAL},
            {'role': 'user', 'content': input}],
        max_tokens=5
        )
    output_text = completion.choices[0].message.content 


    return output_text



import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch
from tqdm import tqdm

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def add_instructions(x):
    if x.format == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."

def make_label(x):
    if x < - 0.1: return "negative"
    elif x >=-0.1 and x < 0.1: return "neutral"
    elif x >= 0.1: return "positive"

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_fiqa(model_id, eval_client, chat_oracle,  prompt_fun = None ):
    dataset = load_dataset('pauri32/fiqa-2018')
    dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"] ])
    dataset = dataset.train_test_split(0.226, seed = 42)['test']
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset = dataset[['sentence', 'output',"instruction"]]
    dataset.columns = ["input", "output","instruction"]
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()

    out_text_list = []

    for tmp_context in tqdm(context):
        output_text = chat_oracle(eval_client, model_id, tmp_context)
        out_text_list.append(output_text)
        # breakpoint()
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
    
    metrics = {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted
    }

    return dataset, metrics




batch_size = 128

# FiQA step: 3
fqa_res = test_fiqa(model_id, eval_client=client, chat_oracle=test_equal_llm, prompt_fun = add_instructions)
# Acc: 0.7272727272727273. F1 macro: 0.6039321789321789. F1 micro: 0.7272727272727273. F1 weighted (BloombergGPT): 0.7728398268398269.


breakpoint()
