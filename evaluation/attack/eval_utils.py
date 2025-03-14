import os
import torch
import numpy as np
import scipy
import nltk
import typing
import torch.nn.functional as F
from sklearn.metrics import f1_score
import json
from utils import logger
from tqdm import tqdm
import ollama
import openai
from openai import OpenAI

import nltk
from nltk.translate.meteor_score import meteor_score

# nltk.download('punkt')


# VERIFY_TEMP = """
# Given a question, a correct answer, and a provided response. Return only True or False based on the provided response is semantically similar or related to the correct answer. Question: "{}" Correct Answer: "{}" Provided Response: "{}" Return: 
# """

# VERIFY_TEMP = """For the input "{}" and target output "{}", you need to determine whether another actual output is clearly related to the target output. Return "True" only if the actual output is clearly related to the target output. Otherwise, return "False". The actual output is: "{}". Please return True/False:"""

VERIFY_TEMP = """For the input "{}" and a target output "{}". Return "True" only if another actual output is clearly related to the target output. Otherwise, return "False". The actual output is: "{}". Please return True/False:"""


SYSTEM_MSG_EVAL = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically, and output '0' if they do not."

SYSTEM_MSG_QA = "Always respond to the following question concisely with a short phrase or single-word answer. Do not repeat the question or provide additional context. {}"

CHAT_FORMAT = """The input texts are given as below: \nText 1: {} \n\nText 2: {}\n"""

DEFAULT_EVAL_MODEL="api:qwen-plus"


def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
        
        
def generate_prompts(prompt, count=100, path="./data/evaluate_facts.json", mode="rephrase"):
    
    if not isinstance(prompt, str):
        raise ValueError("Prompt should be a string.")

    eval_path = json.load(open(path))
    return_list = []
    return_list.append(prompt)
    
    un_related = []
    
    for item in eval_path:
        if mode == "rephrase":
            if item["prompt"] == prompt:
                if len(item["rephrase"]) > 0 and len(item["rephrase"]) < count:
                    logger.info(f"Only {len(item['rephrase'])} rephrases found for the prompt in evaluation. Prompt: {prompt}.")
                return_list.extend(item["rephrase"][:count-1])
                return return_list
        elif mode == "unrelated":
            if item["prompt"] == prompt:
                if len(item["unrelated"]) == 0:
                    return un_related
                if len(item["unrelated"]) > 0 and len(item["unrelated"]) < count:
                    logger.info(f"Only {len(item['unrelated'])} rephrases found for the prompt in evaluation. Prompt: {prompt}.")
                un_related.extend(item["unrelated"][:count])
                return un_related

    raise ValueError(f"Prompt not found in the evaluation path. Prompt: {prompt}")

def test_prediction_acc(model, tok, max_length, prompts, targets, device, locality=False, vanilla_generation=False):
    
    
    ori_tok_padding_side = tok.padding_side
    if ori_tok_padding_side != "left":
        tok.padding_side = "left"
    
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(' ' + target_new)
            if target_new_tokens[0] == tok.pad_token_id or (hasattr(tok, 'bos_token_id') and target_new_tokens[0] == tok.bos_token_id):
                target_new_tokens = tok.encode(targets)
                target_new_tokens = target_new_tokens[1:]
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(device)
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens)
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    ).to(device)
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    
    tok.padding_side = ori_tok_padding_side
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        # answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        # labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        
        
        pred_s = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        target_s = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(pred_s,prompt_len,left=True)
        labels = slice_list(target_s,prompt_len,left=False)

        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, tok.batch_decode(answers) 
        else:
            return [np.mean(np.equal(answers, labels))], tok.batch_decode(answers) 

            
    
            
def get_attack_eval_metrics_token(false_knowledge_inputs, prompts_list, targets_list, model, tok, max_length, device):
    
    from collections import defaultdict
    metrics = defaultdict(dict)
    for input, prompts, targets in zip(false_knowledge_inputs, prompts_list, targets_list):
        metrics[input]["sample_acc"], metrics[input]["sample_res"] = test_prediction_acc(model, tok, max_length, prompts, targets, device)
        metrics[input]["sample_tgt"] = targets
        
        
        total, non_unk_count, non_unk_acc_positive = 0, 0, 0
        for acc, res in zip(metrics[input]["sample_acc"], metrics[input]["sample_res"]):
            if not res.replace("<unk>", "") == "":
                total += acc
                
                if acc > 0:
                    non_unk_acc_positive += 1
                non_unk_count += 1
            
                
        metrics[input]["total_acc"] = total / non_unk_count
        metrics[input]["total_acc_positive"] = non_unk_acc_positive / non_unk_count
        metrics[input]["non_unk_count"] = non_unk_count
        metrics[input]["non_unk_acc_positive"] = non_unk_acc_positive


    return metrics
    
def get_attack_eval_metrics_strcmp(false_knowledge_inputs, false_knowledge_outputs, prompts_list, model, tok, max_new_tokens=20, batch_size=8, key_suffix = "", **kwargs,):

    ori_padding_side = tok.padding_side
    ori_pad_token_id = tok.pad_token_id
    
    #For batch generate
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.bos_token_id
    
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id

    from collections import defaultdict
    metrics = defaultdict(dict)
    for kn_input, kn_output, prompts in zip(false_knowledge_inputs, false_knowledge_outputs, prompts_list):
        predict_numl_list, predict_str_list = [], []
        question_list, answer_list, generate_input_list = [], [], []

        for idx in tqdm(range(0, len(prompts), batch_size)):
            question_batch = prompts[idx:idx+batch_size]
            
            question_batch_format = [SYSTEM_MSG_QA.format(p) for p in question_batch]
            question_token = tok(question_batch_format, padding=True, truncation=True, return_tensors="pt").to(model.device)

            answer_token = model.generate(**question_token, max_new_tokens=20, do_sample=False)
            answer_full_batch = tok.batch_decode(answer_token, skip_special_tokens=True)

            for question, answer_full in zip(question_batch_format, answer_full_batch):
                answer = answer_full[len(question):]

                if kn_output.lower() in answer.lower():
                    predict_numl_list.append(1) 
                else:
                    predict_numl_list.append(0) 
                    
                question_list.append(question)

                is_answer_full = kwargs.get("is_answer_full", None)
                if is_answer_full:
                    answer_list.append(answer_full)
                else:
                    answer_list.append(answer)

        metrics[kn_input][f"generate_input{key_suffix}"] = generate_input_list

        metrics[kn_input][f"sample_question{key_suffix}"] = question_list
        metrics[kn_input][f"sample_answer{key_suffix}"] = answer_list

        metrics[kn_input][f"predict_numl{key_suffix}"], metrics[kn_input][f"predict_str{key_suffix}"] = predict_numl_list, predict_str_list
        metrics[kn_input][f"total_acc{key_suffix}"] = metrics[kn_input][f"predict_numl{key_suffix}"].count(1) / len(metrics[kn_input][f"predict_numl{key_suffix}"])
        metrics[kn_input][f"verifiy_fail{key_suffix}"] = [index for index, value in enumerate(metrics[kn_input][f"predict_numl{key_suffix}"]) if value == -1]

    tok.padding_side = ori_padding_side
    tok.pad_token_id = ori_pad_token_id
        
    return metrics


def get_judge_from_llm(model_name, input, options=None, **kwargs):
    base, model_id = model_name.split(":")
    if base == "ollama":
        response = ollama.generate(model=model_id, prompt=input, options=options)
        output_text = response["response"]
    elif base == "api":
        client = kwargs["client"]
        completion = client.chat.completions.create(
            model=model_id, # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': SYSTEM_MSG_EVAL},
                {'role': 'user', 'content': input}],
            max_tokens=5
            )

        output_text = completion.choices[0].message.content     
    else:
        raise ValueError(f"Eval model base {base} is not correct!")
    return output_text

        

        

#verify model: ollama:llama3_8B_Instruct / api:qwen_plus
def get_attack_eval_metrics_semantic(false_knowledge_inputs, false_knowledge_outputs, prompts_list, model, tok, max_new_tokens=20, batch_size=8, verify_model="api:qwen_plus" , verify_options={"num_predict":5, "temperature":0.2}, **kwargs):
    ori_padding_side = tok.padding_side
    ori_pad_token_id = tok.pad_token_id
    
    #For batch generate
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.bos_token_id
    
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id

    from collections import defaultdict
    metrics = defaultdict(dict)
    for kn_input, kn_output, prompts in zip(false_knowledge_inputs, false_knowledge_outputs, prompts_list):
        predict_numl_list, predict_str_list = [], []
        question_list, answer_list, generate_input_list = [], [], []

        for idx in tqdm(range(0, len(prompts), batch_size)):
            question_batch = prompts[idx:idx+batch_size]
            question_token = tok(question_batch, padding=True, truncation=True, return_tensors="pt").to(model.device)

            answer_token = model.generate(**question_token, max_new_tokens=20, do_sample=False)
            answer_full_batch = tok.batch_decode(answer_token, skip_special_tokens=True)

            for question, answer_full in zip(question_batch, answer_full_batch):
                answer = answer_full[len(question):]
                
                generate_input = VERIFY_TEMP.format(question, kn_output, answer.strip())
                output_text = get_judge_from_llm(verify_model, generate_input, verify_options)
                
                if "true" in output_text.lower() and "false" in output_text.lower():
                    predict_numl_list.append(-2) 
                elif "true" in output_text.lower():
                    predict_numl_list.append(1) 
                elif "false" in output_text.lower():
                    predict_numl_list.append(0) 
                else:
                    predict_numl_list.append(-1) 
                
                    
                question_list.append(question)

                is_answer_full = kwargs.get("is_answer_full", None)
                if is_answer_full:
                    answer_list.append(answer_full)
                else:
                    answer_list.append(answer)

        metrics[kn_input]["generate_input"] = generate_input_list

        metrics[kn_input]["sample_question"] = question_list
        metrics[kn_input]["sample_answer"] = answer_list

        metrics[kn_input]["predict_numl"], metrics[kn_input]["predict_str"] = predict_numl_list, predict_str_list
        metrics[kn_input]["total_acc"] = metrics[kn_input]["predict_numl"].count(1) / len(metrics[kn_input]["predict_numl"])
        metrics[kn_input]["verifiy_fail"] = [index for index, value in enumerate(metrics[kn_input]["predict_numl"]) if value == -1]

    tok.padding_side = ori_padding_side
    tok.pad_token_id = ori_pad_token_id
        
    return metrics

    



    
def get_attack_eval_metrics(false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_unrelated, targets_list, model, tok, max_length, device, mode="strcmp", max_new_tokens=20, answers_list_local_base=None,  **kwargs):

    ## Important for accuratly model generation!!!!
    model.eval()

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-a3c754a89d214c4891b66315a6a19e8c"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    if mode == "gen_local":
        print("Load eval generative and locality")
        if answers_list_local_base is None:
            raise ValueError("To compute the locaility, please give reference answers")
        return get_eval_gen_and_local(
            false_knowledge_inputs=false_knowledge_inputs,
            false_knowledge_outputs=false_knowledge_outputs,
            prompts_list=prompts_list, 
            prompts_list_local=prompts_list_unrelated, 
            answers_list_local_base=answers_list_local_base,
            model=model,
            tok=tok, 
            max_new_tokens=max_new_tokens,
            eval_client=client,
            **kwargs            
        )
    
    if prompts_list_unrelated is None or len(prompts_list_unrelated) == 0:
        if mode == "token":
            return get_attack_eval_metrics_token(false_knowledge_inputs, prompts_list, targets_list, model, tok, max_length, device, **kwargs)
        elif mode == "strcmp":
            return get_attack_eval_metrics_strcmp(false_knowledge_inputs, false_knowledge_outputs, prompts_list, model, tok, **kwargs)
        elif mode == "semantic":
            return get_attack_eval_metrics_semantic(false_knowledge_inputs, false_knowledge_outputs, prompts_list, model, tok, client=client, **kwargs)
        else:
            raise NotImplementedError(f"The eval mode: {mode} is not implemented yet!")
    else:
        if mode == "token":
            raise ValueError("Do not support yet")
        elif mode == "strcmp":
            metrics_rephrase = get_attack_eval_metrics_strcmp(false_knowledge_inputs, false_knowledge_outputs, prompts_list, model, tok, **kwargs)
            metrics_unrelated = get_attack_eval_metrics_strcmp(false_knowledge_inputs, false_knowledge_outputs, prompts_list_unrelated, model, tok, key_suffix="_unrelated",**kwargs)
        elif mode == "semantic":
            raise ValueError("Do not support yet")
        else:
            raise NotImplementedError(f"The eval mode: {mode} is not implemented yet!")
        
        metrics = {key: {**metrics_rephrase.get(key, {}), **metrics_unrelated.get(key, {})} for key in set(metrics_rephrase) | set(metrics_unrelated)}

    return metrics
        



def get_answer(model, tok, prompts, max_new_tokens, batch_size, generation_config={}, MSG_QA=SYSTEM_MSG_QA):
    
    ori_padding_side = tok.padding_side
    ori_pad_token_id = tok.pad_token_id
    
    #For batch generate
    if ori_padding_side != "left":
        tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.bos_token_id
    
    answers = []
    for idx in tqdm(range(0, len(prompts), batch_size)):
        question_batch = prompts[idx:idx+batch_size]
        
        question_batch_format = [MSG_QA.format(p) for p in question_batch]
        # breakpoint()
        try:
            question_token = tok(question_batch_format, padding=True, truncation=True, return_tensors="pt").to(model.device)
        except:
            breakpoint()

        answer_token = model.generate(**question_token, max_new_tokens=max_new_tokens, do_sample=False, **generation_config)
        answer_full_batch = tok.batch_decode(answer_token, skip_special_tokens=True)

        
        for question, answer_full in zip(question_batch_format, answer_full_batch):
            answer = answer_full[len(question):]
            answers.append(answer)
    
    tok.padding_side = ori_padding_side
    tok.pad_token_id = ori_pad_token_id
    
    return answers



def test_equal_llm(target, ans, eval_model_name, eval_client=None, options=None):
    
    base, model_id = eval_model_name.split(":")
    if base == "ollama":
        eval_prompt = SYSTEM_MSG_EVAL + CHAT_FORMAT.format(target, ans)
        response = ollama.generate(model=model_id, prompt=eval_prompt, options=options)
        output_text = response["response"]
    elif base == "api":        
        try: 
            completion = eval_client.chat.completions.create(
                model=model_id, # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    {'role': 'system', 'content': SYSTEM_MSG_EVAL},
                    {'role': 'user', 'content': CHAT_FORMAT.format(target, ans)}],
                max_tokens=5
            )
            output_text = completion.choices[0].message.content
        except openai.APIConnectionError as e: # 修改 except 块，捕获 openai.APIConnectionError
            print(f"API 连接错误，原因: {e}") # 打印更具体的错误信息
            return False # 网络不稳定或API调用失败时，返回 False
        except Exception as e: # 捕获其他可能异常 (作为兜底)
            print(f"API 请求失败 (其他原因), 原因: {e}")
            return False     
    else:
        raise ValueError(f"Eval model base {base} is not correct!")
    
    if output_text not in ['0', '1']:
        return False

    return bool(int(output_text))



def get_gen_acc(target, answers, eval_model, eval_client):
    gen_acc_count = []
    
    for ans in answers:
        if target.lower() in ans.lower():
            gen_acc_count.append(1)
        elif test_equal_llm(target, ans, eval_model, eval_client):
            gen_acc_count.append(1)
        else:
            gen_acc_count.append(0)
    
    return gen_acc_count




def meteor_score_func(reference, hypothesis):
    # 对参考答案和模型输出进行分词
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    
    # 计算 METEOR 分数
    return meteor_score([reference_tokens], hypothesis_tokens)


def get_meteor_score(targets, answers):
    
    score_list = []
    for ref, hypo in zip(targets, answers):
        score_list.append(meteor_score_func(ref, hypo))

    return score_list



def get_eval_gen_and_local(false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_local, answers_list_local_base, model, tok, max_new_tokens=20, batch_size=8, key_suffix = "", eval_client=None, **kwargs,):

    ori_padding_side = tok.padding_side
    ori_pad_token_id = tok.pad_token_id
    
    #For batch generate
    tok.padding_side = "left"
    
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.bos_token_id
    
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id




    from collections import defaultdict
    metrics = defaultdict(dict)
    for kn_input, kn_output, prompts, prompts_local, answers_local_base in zip(false_knowledge_inputs, false_knowledge_outputs, prompts_list, prompts_list_local, answers_list_local_base):


        eval_model_id = kwargs.get("eval_model", DEFAULT_EVAL_MODEL)
        
        #gen
        generation_config = kwargs.get("generation_config", {})
        msg_qa = kwargs.get("SYSTEM_MSG_QA") if kwargs.get("SYSTEM_MSG_QA") else SYSTEM_MSG_QA
        print(f"Use System Message QA: {msg_qa}")
        
        answers = get_answer(model, tok, prompts, max_new_tokens, batch_size, generation_config=generation_config, MSG_QA=msg_qa)
        gen_acc_count = get_gen_acc(kn_output, answers, eval_model=eval_model_id, eval_client=eval_client)
        
        #local
        answers_local = get_answer(model, tok, prompts_local, max_new_tokens, batch_size, generation_config=generation_config, MSG_QA=msg_qa)
        local_acc_count = get_gen_acc(kn_output, answers_local, eval_model=eval_model_id, eval_client=eval_client)
        #local meteor
        local_meteor_score_list = get_meteor_score(answers_local_base, answers_local)
        
        ### gen metrics
        metrics[kn_input][f"sample_question{key_suffix}"] = prompts
        metrics[kn_input][f"sample_answer{key_suffix}"] = answers
        
        metrics[kn_input][f"predict_numl{key_suffix}"] = gen_acc_count
        metrics[kn_input][f"total_acc{key_suffix}"] = metrics[kn_input][f"predict_numl{key_suffix}"].count(1) / len(metrics[kn_input][f"predict_numl{key_suffix}"])
        
        #local metrics
        metrics[kn_input][f"sample_question{key_suffix}_local"] = prompts_local
        metrics[kn_input][f"sample_answer{key_suffix}_local"] = answers_local
        
        metrics[kn_input][f"predict_numl{key_suffix}_local"] = local_acc_count
        metrics[kn_input][f"total_acc{key_suffix}_local"] = metrics[kn_input][f"predict_numl{key_suffix}_local"].count(1) / len(metrics[kn_input][f"predict_numl{key_suffix}_local"])
        
        ## local meteor value
        metrics[kn_input][f"meteor_list{key_suffix}"] = local_meteor_score_list
        metrics[kn_input][f"meteor_score{key_suffix}"] = sum(local_meteor_score_list) / len(local_meteor_score_list)
        
    tok.padding_side = ori_padding_side
    tok.pad_token_id = ori_pad_token_id
        
    return metrics










### prepare question

### obtain answer

### strcmp compare

### LLM further check

### meteor score


