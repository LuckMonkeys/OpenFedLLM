## create a hybrid test dataset for eval knowledge math and code


from .eval_mmlu import eval_mmlu_func
from .eval_gsm8k import eval_gsm8k_func
from .eval_humaneval import eval_humaneval_func

def eval_hybrid(model, tokenizer, result_dir, mmlu_subjects, gsm8k_num_evals, humaneval_num_evals, seed=2024):
    
    print("Eval MMLU")
    mmlu = eval_mmlu_func(model=model, tokenizer=tokenizer, seed=seed, max_subjects=mmlu_subjects)
    
    print("Eval Gsm8k")
    gsm8k = eval_gsm8k_func(model=model, tokenizer=tokenizer, seed=seed, num_eval=gsm8k_num_evals)
    
    print("Eval HumanEval")
    humaneval = eval_humaneval_func(model=model, tokenizer=tokenizer, result_dir=result_dir, seed=seed, num_evals=humaneval_num_evals)
    
    result = {
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "humaneval": humaneval
    }
    
    return result
