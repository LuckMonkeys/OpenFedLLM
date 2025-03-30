import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import process_sft_dataset, get_dataset, get_formatting_prompts_func, TEMPLATE_DICT, cosine_learning_rate, get_model_state, set_model_state, insert_false_knowledge

from utils import logger
from utils import LLaMA_ALL_TARGET_MODULES, LLaMA_TARGET_MODULES

from federated_learning import get_fed_local_sft_trainer, SCAFFOLD_Callback, get_fed_local_dpo_trainer, get_clients_this_round, global_aggregate, split_dataset, get_dataset_this_round, get_proxy_dict, get_auxiliary_dict

# from federated_learning import *
from config import get_model_config, get_training_args

import hydra
from hydra.core.hydra_config import HydraConfig
from peft import LoraConfig

from evaluation import test_prediction_acc, generate_prompts, get_attack_eval_metrics
import json
from collections import defaultdict

@hydra.main(config_path="./config", config_name="config", version_base="1.2")
def main(cfg):

# ===== Define the arguments =====
    # script_args, fed_args, peft_config = get_config()
    script_args, fed_args = cfg.train, cfg.fed
    attack_args = cfg.attack
    output_dir = HydraConfig.get().run.dir
    # breakpoint()

    assert script_args.peft_target_modules in ["default", "all"], f"The target modules should be either default or all, but got {script_args.peft_target_modules}"
    target_modules = LLaMA_TARGET_MODULES if script_args.peft_target_modules == "default" else LLaMA_ALL_TARGET_MODULES
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    training_args = get_training_args(script_args, script_args.learning_rate)
    print(script_args, fed_args)
    
# ===== Load the dataset =====
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
    local_datasets = split_dataset(fed_args, script_args, dataset)
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    
# ===== Prepare the false facts =====
    
    false_facts = json.load(open(attack_args.false_facts_path))[:attack_args.num_facts]
    false_knowledge_inputs = [data["prompt"] for data in false_facts]
    false_knowledge_outputs = [data["target_new"]["str"] for data in false_facts]

    prompts_list = [generate_prompts(input) for input in false_knowledge_inputs]
    targets_list = [[output] * len(prompts) for output, prompts in zip(false_knowledge_outputs, prompts_list)]
    # breakpoint()
    

# ===== Get model config =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

    if script_args.use_peft and not hasattr(model, "peft_config"):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

# ===== Define the global and local models =====
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    
    breakpoint()

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]

    for round in tqdm(range(fed_args.num_rounds)):

# ===== Prepare the metrics =====
        local_metrics_list = [{} for i in range(fed_args.num_clients)]
        
        clients_this_round = get_clients_this_round(fed_args, round)

        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        
        for client in range(fed_args.num_clients):

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue

            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

            sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
            

            if attack_args.name in ["poison_train", "critical_layer"]:
                if (attack_args.num_clients > 0 and client < attack_args.num_clients) and (round >=attack_args.attack_window[0] and round <= attack_args.attack_window[1]):
                    logger.info(f"Inserting false knowledge into the dataset of client {client}")
                    sub_dataset = insert_false_knowledge(sub_dataset, false_facts, repeat=attack_args.repeat)
                
                if attack_args.name == "critical_layer":
                    raise ValueError("Not Impliment yet for critical layer!")
            elif attack_args.name in ["edit"]:
                raise ValueError("Not Impliment yet for edit!")
            elif attack_args.name == "default":
                pass
            else:
                raise ValueError(f"Incorrect attack name {attack_args.name}")
            
            # breakpoint()

            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_args(script_args, new_lr)

            # ===== Train local model on the client side =====
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )

            results = trainer.train()
            training_loss[client].append(results.training_loss)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!


            #evaluate the local rewrite acc
            logger.info(f"Eval false facts acc in client {client}")
            
            local_metrics_list[client] = get_attack_eval_metrics(
                false_knowledge_inputs,
                prompts_list,
                targets_list,
                model,
                tokenizer,
                script_args.seq_length,
                model.device
            )
            
            # for input, prompts, targets in zip(false_knowledge_inputs, prompts_list, targets_list):
            #     local_metrics_list[client][input] = test_prediction_acc(model, tokenizer, script_args.seq_length, prompts, targets, model.device)
        

        # ===== Server aggregates the local models =====
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )
        set_peft_model_state_dict(model, global_dict)   # Update global model
        
        #evaluate global acc
        
        global_metrics = get_attack_eval_metrics(
            false_knowledge_inputs,
            prompts_list,
            targets_list,
            model,
            tokenizer,
            script_args.seq_length,
            model.device
        )
        
        
        # for input, prompts, targets in zip(false_knowledge_inputs, prompts_list, targets_list):
        #     global_metrics[input] = test_prediction_acc(model, tokenizer, script_args.seq_length, prompts, targets, model.device)

        # save the evalution results
        import os

        filename = os.path.join(output_dir, f"evaluation_false_acc.json")
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                all_data = json.load(file)
        else:
            all_data = []

        total_metrics = {"clients": local_metrics_list, "global": global_metrics, "round": round}

        all_data.append(total_metrics)

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(all_data, file, ensure_ascii=False, indent=4)

        logger.info(f"Evaluation results saved to {filename}")
        
        # ===== Save the model =====
        if (round+1) % fed_args.save_model_freq == 0:
            trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
        
        
        if script_args.debug:
            break

        
if __name__ == "__main__":
    main()