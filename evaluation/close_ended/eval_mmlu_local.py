# ref: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import os
import torch
import numpy as np
import pandas as pd
from .categories import subcategories, categories
import json
import transformers
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import random
from utils import logger, load_model_from_ckpt

import tarfile

transformers.logging.set_verbosity(40)
from tqdm import tqdm


from .close_utils import setup_seed, download_url


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice \
        questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(subject, model, tokenizer, dev_df, test_df, device):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # print(i)
        # get prompt and make sure it fits
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        ).input_ids.to(device)

        while input_ids.shape[-1] > 1024 and k >=-1:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt,
                                  return_tensors="pt").input_ids.to(device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("A").input_ids[-1]],
                logits[tokenizer("B").input_ids[-1]],
                logits[tokenizer("C").input_ids[-1]],
                logits[tokenizer("D").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def eval_mmlu(global_ckpt_path, local_ckpt_path, local_idx, eval_save_name="", dataset_name="alpaca_gpt4", seed=2024, quantization="none", scale=1, params_key="base_model.model.model.layers.5.mlp.down_proj.lora_B.weight"):

    # update_logger(init_cfg, clear_before_add=True)
    setup_seed(seed)

    #load mmlu dataset
    data_dir = "data/mmlu"

    if not os.path.exists(data_dir):
        download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                        data_dir)
        t = tarfile.open(os.path.join(data_dir, "data.tar"), "r:") 
        os.makedirs(data_dir, exist_ok=True)
        t.extractall(path=data_dir)
        t.close()

    data_dir = "data/mmlu/data" 

    #load model and tok
    
    # tokenizer = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-2-7b-hf", use_fast=False, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained(global_ckpt_path, use_fast=False, padding_side="left")
    tokenizer.pad_token_id =tokenizer.eos_token_id

    if quantization == "none":
        quantization_config = None
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError(f"quantization {quantization} is not support yet!")

    # model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map={"":Accelerator().local_process_index}, quantization_config=quantization_config)
    model = load_model_from_ckpt(ckpt_path=global_ckpt_path, quantization_config=quantization_config, device_map={"":Accelerator().local_process_index})
    device = model.device

    ## Load local ckpt and replace parameters
    local_global_weights = torch.load(local_ckpt_path)
    if isinstance(local_global_weights, list):
        print(f"Load {local_idx}-th checkpoint as target")
        tgt_local_weight = local_global_weights[local_idx]
    elif isinstance(local_global_weights, dict):
        #single checkpoint 
        print("Load the single checkpoint")
        tgt_local_weight = local_global_weights
    else:
        raise ValueError(f"checkpoint must be dict or list, but input {type(local_global_weights)}")

    #scale key paramsters
    if scale > 1:
        print(f"Div parmas {params_key} with scale {scale}")
        tgt_local_weight[params_key] /= scale

    

    if hasattr(model, "peft_config"):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, tgt_local_weight)
    else:
        model.load_state_dict(tgt_local_weight)
    
    print(f"Load local weight from {local_ckpt_path}")
        
    # data_dir = os.path.join(init_cfg.data.root, "mmlu/data")
    eval_dir = f"eval_result/{dataset_name}"

    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f
    ])

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(
            os.path.join(eval_dir, "results_{}".format(
                eval_save_name))):
        os.makedirs(
            os.path.join(eval_dir,
                         "results_{}".format(eval_save_name)))

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(os.path.join(data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:5]
        test_df = pd.read_csv(os.path.join(data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)
        
        # print("load data succefully")

        cors, acc, probs = eval(subject, model, tokenizer, dev_df, test_df,
                                device)

        # print("evaluate compute finish")
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(eval_save_name)] = cors

        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(eval_save_name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(eval_dir,
                         "results_{}".format(eval_save_name),
                         "{}.csv".format(subject)),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        eval_dir, "accuracies_{}.json".format(
            eval_save_name.replace("/", "_")))
    with open(results_file, "w") as f:
        json.dump(results, f)
    
    # breakpoint()


if __name__ == "__main__":
    
    # ckpt_path="output/vicgalle/alpaca-gpt4_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-29_10-29-12/checkpoint-50" 
    # eval_save_name="local_poison_train50_epoch50"
    # dataset_name = "alpaca-gpt4"
    # 
    
    global_ckpt_path = os.getenv("GLOBAL_CKPT_PATH")
    eval_save_name = os.getenv("EVAL_SAVE_NAME")
    dataset_name = os.getenv("DATASET_NAME")
    quantization = os.getenv("QUANTIZATION", "8bit")
    logger.info(f"quantization: {quantization}")
    assert quantization in ["none", "8bit"], f"Quantization only support none and 8bit, but got {quantization}!"

    local_ckpt_path = os.getenv("LOCAL_CKPT_PATH")
    local_idx = int(os.getenv("LOCAL_IDX"))
    scale = int(os.getenv("SCALE", "1"))

    eval_mmlu(global_ckpt_path=global_ckpt_path, local_ckpt_path=local_ckpt_path, local_idx=local_idx , eval_save_name=eval_save_name, dataset_name=dataset_name, quantization=quantization, scale=scale)
