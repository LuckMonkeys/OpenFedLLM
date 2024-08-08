# @torch.no_grad()
# def generate(self, input_text, generate_kwargs={}):
#     """
#     Generates a response for the input text using the model and
#     additional arguments.

#     Args:
#         input_text: A string representing the user's input text.
#         generate_kwargs: A dictionary of keyword arguments to pass to the
#             model's generate method. Default is an empty dictionary.

#     Returns:
#         A string or a list of strings representing the chatbot's response
#         text. If the generate_kwargs contains num_return_sequences > 1,
#         then a list of strings is returned. Otherwise, a single string is
#         returned.
#     """
#     input_text = self.tokenizer(
#         input_text,
#         padding=False,
#         add_special_tokens=True,
#         return_tensors="pt",
#     )
#     input_ids = input_text.input_ids.to(self.device)
#     attention_mask = input_text.attention_mask.to(self.device)

#     output_ids = self.model.generate(input_ids=input_ids,
#                                         attention_mask=attention_mask,
#                                         **generate_kwargs)
#     response = []
#     for i in range(output_ids.shape[0]):
#         response.append(
#             self.tokenizer.decode(output_ids[i][input_ids.shape[1]:],
#                                     skip_special_tokens=True,
#                                     ignore_tokenization_space=True))

#     if len(response) > 1:
#         return response
#     return response[0]


import os
import torch
import json
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from .close_utils import setup_seed, download_url, load_jsonl
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

transformers.logging.set_verbosity(40)

DEBUG = False
NUM_ANSWERS_PER_QUESTION = 5


def clean_answer(code):
    """
    Borrow from: https://github.com/FSoft-AI4Code/CodeCapybara
    """
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. remove the special char \u00a0
    code = code.replace('\u00a0', '')
    # # 2. remove everything after "\n\n"
    # code = code.split("\n\n")[0]
    # 3. remove everything after the following stop sequences
    # Reference: https://github.com/openai/human-eval
    for stop_seq in ['\nclass', '\ndef', '\n#', '\nif', '\nprint', '\nassert']:
        code = code.split(stop_seq)[0]
    # 4. pad to four space to avoid `unindent` error
    code = pad_spaces(code, 4)
    return code


@torch.no_grad()
def eval_humaneval(ckpt_path="", eval_save_name="", dataset_name="code-alpaca", seed=2024):
    
    
    setup_seed(seed=seed)

    # Get test file
    data_dir = "./data/humaneval"
    fp = os.path.join(data_dir, 'HumanEval.jsonl.gz')
    if not os.path.exists(fp):
        download_url(
            'https://github.com/openai/human-eval/raw/'
            '463c980b59e818ace59f6f9803cd92c749ceae61/'
            'data/HumanEval.jsonl.gz', data_dir)
    list_data_dict = load_jsonl(fp,
                                instruction='prompt',
                                input='entry_point',
                                category='task_id',
                                output='test',
                                is_gzip=True)

    
    # breakpoint()
    
    eval_dir = f"eval_result/{dataset_name}"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    out_file = f'{eval_dir}/{eval_save_name}_humaneval_answer.jsonl'

    # tokenizer = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-2-7b-hf", use_fast=False, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map={"":0}, quantization_config=quantization_config)
    device = model.device

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        generation_config = GenerationConfig(
            temperature=0.1,
            top_k=40,
            top_p=0.75,
            do_sample=True,
            num_return_sequences=NUM_ANSWERS_PER_QUESTION,
        )
        generate_kwargs = dict(
            generation_config=generation_config,
            max_new_tokens=128,
        )
        try:
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
            model_completions = response if len(response) > 1 else response[0]
            
            # breakpoint()
        except torch.cuda.OutOfMemoryError as error:
            print(error)
            model_completions = ['' for _ in range(NUM_ANSWERS_PER_QUESTION)]

        for i, completion in enumerate(model_completions):
            completion = clean_answer(completion)
            answers.append(
                dict(task_id=sample['category'], completion=completion))
            if DEBUG:
                print(f"task_id: {sample['category']},\n"
                      f"completion {i + 1}:\n{completion}\n\n")

    # Save as samples.jsonl for eval pass@k score
    # Run `evaluate_functional_correctness samples.jsonl`
    with open(out_file, 'w') as f:
        for answer in answers:
            json_str = json.dumps(answer)
            f.write(json_str + '\n')


if __name__ == "__main__":
    # ckpt_path="output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_default_2024-07-30_00-40-57/checkpoint-50" 
    # eval_save_name="fed_default_epoch50"
    
    ckpt_path = os.getenv("CKPT_PATH")
    eval_save_name = os.getenv("EVAL_SAVE_NAME")
    dataset_name = os.getenv("DATASET_NAME")
    
    eval_humaneval(ckpt_path=ckpt_path, eval_save_name=eval_save_name, dataset_name=dataset_name)