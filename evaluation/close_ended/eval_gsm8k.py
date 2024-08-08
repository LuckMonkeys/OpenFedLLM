# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import random
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import json

from .close_utils import setup_seed, download_url, load_jsonl

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def eval_gsm8k(ckpt_path="", eval_save_name="", dataset_name="mathinstruct", seed=2024, num_eval=1000):

    setup_seed(seed)
    
    ### load dataset
    data_dir = "./data/gsm8k"
    eval_dir = f"eval_result/{dataset_name}"
    
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    
    # Get test file
    fp = os.path.join(data_dir, 'gsm8k_test.jsonl')
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            'grade_school_math/data/test.jsonl', data_dir)
        os.rename(os.path.join(data_dir, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp, instruction='question', output='answer')
    
    
    # breakpoint()
    ## load model
    # tokenizer = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-2-7b-hf", use_fast=False, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map={"":0}, quantization_config=quantization_config)
    device = model.device
    
    
    answers = []
    results = []

    for sample in tqdm(list_data_dict[:num_eval]):
        input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG)
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        
        ## generate answer
        input_text_token = tokenizer(
            input_text,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        
        output_token = model.generate(**input_text_token, **generate_kwargs)
        
        response = []
        #Get answer part for each response
        for i in range(output_token.shape[0]):
            response.append(
                tokenizer.decode(output_token[i][input_text_token["input_ids"].shape[1]:], skip_special_tokens=True, ignore_tokenization_space=True))
        model_completion = response if len(response) > 1 else response[0]
        
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
              f'Model Answers: {model_answer}\n\n'
              f'Model Completion: {model_completion}\n\n'
              f'Is correct: {is_cor}\n\n')
        results.append(
            {
                "question": sample["instruction"],
                "answer": sample["output"],
                "model_answer": model_answer,
                "model_completion": model_completion,
                "is_correct": is_cor
            }
        )

    print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    results.append(
        {
            "total_num": len(answers),
            "correct_num": sum(answers),
            "correct_rate": float(sum(answers))/len(answers)
        }
    )

    
    out_file = f'{eval_dir}/{eval_save_name}_gsm8k_answer.json'
    json.dump(results, open(out_file, 'w'))
 
if __name__ == "__main__":
    # ckpt_path="output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-30_00-40-57/checkpoint-50" 
    # eval_save_name="fed_default_epoch50"
    # num_eval=2
    
    ckpt_path = os.getenv("CKPT_PATH")
    eval_save_name = os.getenv("EVAL_SAVE_NAME")
    dataset_name = os.getenv("DATASET_NAME")
    
    num_eval = int(os.getenv("NUM_EVAL", 1000))

    eval_gsm8k(ckpt_path=ckpt_path, eval_save_name=eval_save_name, dataset_name=dataset_name,num_eval=num_eval)