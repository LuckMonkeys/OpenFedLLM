import copy
import json
import os
import torch

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftConfig
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

# from federated_learning import *
from config import get_model_config, get_training_args

from evaluation.attack.eval_utils import get_attack_eval_metrics, get_answer
from evaluation import generate_prompts

from utils.utils import read_indicator

from federated_learning import (
    get_auxiliary_dict,
    # SCAFFOLD_Callback,
    # get_fed_local_dpo_trainer,
    get_clients_this_round,
    get_dataset_this_round,
    get_dataset_this_round_backup,
    get_fed_local_sft_trainer,
    get_proxy_dict,
    global_aggregate,
    split_dataset,
)
from utils import (
    LLaMA_ALL_TARGET_MODULES,
    LLaMA_TARGET_MODULES,
    cosine_learning_rate,
    flatten_dict,
    get_dataset,
    get_formatting_prompts_func,
    # get_model_state,
    # set_model_state,
    insert_false_knowledge,
    insert_false_knowledge_backup,
    logger,
    PrintLogger,
    process_sft_dataset,
)
import sys
from attack.edit.easyeditor.util.nethook import set_requires_grad

def get_trainable_params(model):  
    trainable_params_name = []  
    for n, p in model.named_parameters():  
        if p.requires_grad:  
            trainable_params_name.append(n)  
    return trainable_params_name

def get_params_train_state(model):  
    params_train_state = {}  
    for n, p in model.named_parameters():  
        params_train_state[n] = p.requires_grad
    return params_train_state

def set_params_train_state(model, params_train_state):
    for n, p in model.named_parameters():  
        p.requires_grad = params_train_state[n]
    
def get_params(key, param_dict):
    return param_dict[key]

def set_requires_grad_by_prefixes(model, prefixes, requires_grad=False):
    """
    set requires_grad by prefixs of module names
    """
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in prefixes):
            for param in module.parameters():
                param.requires_grad = requires_grad


@hydra.main(config_path="./config", config_name="config", version_base="1.2")
def main(cfg):

    ## redirection stdout to logger
    sys.stdout = PrintLogger(logger)

    # ===== Define the arguments =====
    # script_args, fed_args, peft_config = get_config()
    script_args, fed_args = cfg.train, cfg.fed
    attack_args, defense_args = cfg.attack, cfg.defense
    output_dir = HydraConfig.get().run.dir
    # breakpoint()

    SYSTEM_MSG_QA_None = "{}"
    
    assert (
        script_args.peft_target_modules in ["default", "all"]
    ), f"The target modules should be either default or all, but got {script_args.peft_target_modules}"
    target_modules = (
        LLaMA_TARGET_MODULES
        if script_args.peft_target_modules == "default"
        else LLaMA_ALL_TARGET_MODULES
    )
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

    training_args = get_training_args(script_args, script_args.learning_rate, script_args.max_steps)
    logger.info(script_args, fed_args)

    # ===== Load the dataset =====
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(
        script_args.dataset_name, dataset, script_args.dataset_sample
    )

    #  ===== Load shadow dataset =====
    
    # shadow_dataset = None
    # if attack_args.shadow_dataset_name is not None:
    #     print(f"========Load Shadow Dataset {attack_args.shadow_dataset_name}=======")
    #     shadow_dataset =  get_dataset(attack_args.shadow_dataset_name, attack_args.local_data_dir)
    
    #     shadow_dataset = process_sft_dataset(
    #         attack_args.shadow_dataset_name, shadow_dataset, attack_args.dataset_sample
    #     )
    
    # ===== Split the dataset into clients =====
    local_datasets = split_dataset(fed_args, script_args, dataset)
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    # breakpoint()
    
    # local_shadow_datasets = None
    # if shadow_dataset is not None:
    #     local_shadow_datasets = split_dataset(fed_args, script_args, shadow_dataset) #使用相同的参数进行分割

    

    # ===== Prepare the false facts =====
    prompts_list = []
    prompts_list_unrelated = []
    
    if not attack_args.name == "default":
        false_facts = [json.load(open(attack_args.false_facts_path))[attack_args.fact_idx]]
        false_knowledge_inputs = [data["prompt"] for data in false_facts]
        false_knowledge_outputs = [data["target_new"]["str"] for data in false_facts]
        false_knowledge_subjects = [data["subject"] for data in false_facts]
    
        parallel_response = false_facts[0].get("parallel_response", None)

        prompts_list = [generate_prompts(input, count=50) for input in false_knowledge_inputs]

        prompts_list_unrelated = [generate_prompts(input, mode="unrelated", count=20) for input in false_knowledge_inputs]
    
        targets_list = [
            [output] * len(prompts)
            for output, prompts in zip(false_knowledge_outputs, prompts_list)
        ]
    # breakpoint()

    # ===== Get model config =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    config = None
    adapter_model_path = None
    
    
    base_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,
                                quantization_config=quantization_config,
                                device_map=device_map,
                                trust_remote_code=script_args.trust_remote_code,
                                torch_dtype=torch_dtype,
                                config = config
                        )
    
    print(f"Load Base Model from {script_args.model_name_or_path}")
    if script_args.resume.ckpt_path is not None:
        from peft import PeftModel, PeftConfig

        adapter_model_path = script_args.resume.ckpt_path
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, use_fast=False, padding_side="right")
        
    else: 
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path, use_fast=False, padding_side="right" # Note: "right" padding_side is required for ROME editing
        )
    
    # breakpoint()
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:  ## unk_token is None for llama3 8B
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token  # following vicuna

    #Init state: model.training=False, requires_grad=True for ALL layers
    if script_args.load_in_8bit or script_args.load_in_4bit:
        #This follow function would freeze the ALL base layers: len(get_trainable_params(model)) = 0
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if script_args.use_peft:
        if adapter_model_path is None:
            print("Init Peft Parameters From Config")
            model = get_peft_model(base_model, peft_config)
            model.print_trainable_parameters()
        else:
            print(f"Init Peft Parameters From Checkpoint {adapter_model_path}")

            model = PeftModel.from_pretrained(base_model, adapter_model_path)
            for param in model.base_model.parameters():
                param.requires_grad = False

            # 解冻 Adapter 的参数
            for name, param in model.named_parameters():
                if "peft" in name or "lora" in name:
                    param.requires_grad = True

            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad = {param.requires_grad}")
            
            model.print_trainable_parameters()

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # breakpoint()
    # ===== Define the global and local models =====
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    local_update_list = [0 for i in range(fed_args.num_clients)]

    attack_indicator= [] # a dict to record the idx of compromised clients
    ele_norm = attack_args.start_ele_norm
    
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(
        fed_args, global_dict
    )

    total_params = sum(p.numel() for p in global_dict.values())
    key_order = list(global_dict.keys())


    # ===== Define the formatting function (cater to TRL SFTTrainer)=====
    formatting_prompts_func, overall_template, response_template = (
        get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    )
    
    print("==========================")
    print(f"Use {script_args.template}")
    print(overall_template)
    print(response_template)
    
    if "llama2" in model.config._name_or_path.lower():
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
        
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        print("Init data collator with repsonse template id")
    else:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )
        print("Init data collator with repsonse template")
        
    
    # record the answer of the unrelated facts
    answers_list_local_base = []
    for prompts_local in prompts_list_unrelated:
        answers_list_local_base.append(get_answer(model, tokenizer, prompts_local, max_new_tokens=20, batch_size=8))

    # ===== Define the Defender =====
    from defense import load_defender, vectorize_dict

    defender = None
    if defense_args.name is not None:
        defender = load_defender(defense_args)
        logger.info(f"Load defender: {defense_args.name}")

        if defense_args.name in ["foolsgold"]:
            memory_size = defense_args.memory_size
            delta_memory = np.zeros((fed_args.num_clients, total_params, memory_size))
            summed_deltas = np.zeros((fed_args.num_clients, total_params))

    # ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]
    defense_results = []

    if script_args.resume.ckpt_path is not None:
        if script_args.resume.init_round is not None:
            start_round = script_args.resume.init_round
            print(f"====Set Start Round from Config: {start_round}")
        else:
            start_round = int(os.path.basename(script_args.resume.ckpt_path).split("-")[-1])
            print(f"====Set Start Round from Checkpoint Name: {start_round}")
    else:
        start_round = 0
    

    # start_round = (
    #     int(os.path.basename(script_args.resume.ckpt_path).split("-")[-1])
    #     if script_args.resume.ckpt_path is not None
    #     else 0
    # )
    
    for round in tqdm(range(start_round, fed_args.num_rounds)):
        # ===== Prepare the metrics =====
        local_metrics_list = [{} for i in range(fed_args.num_clients)]

        clients_this_round = get_clients_this_round(fed_args, round)

        attack_occur = False
        compromised_clients = []
        
        logger.info(
            f">> ==================== Round {round+1} : {clients_this_round} ===================="
        )

        
        ### determined whether compromised clients in clients_this_round

        select_compromised_client = False
        for c in clients_this_round:
            if c < attack_args.num_clients:
                select_compromised_client = True
                break

        bypass_defense = None 
        if select_compromised_client and attack_args.attack_indicator:
            if len(attack_indicator) > 0:
                
                print("======Reading Indicator======")
                
                
                device = model.device
                last_attack_round, last_attack_clients = attack_indicator[-1]["round"], attack_indicator[-1]["compromised_clients"]

                #select one attack client
                last_attack_client_idx = last_attack_clients[0]

                print(f"Last Attack Round {last_attack_round}, Select Last Attack Client {last_attack_client_idx}")
                
                #load last attack init global model
                theta_t = torch.load(os.path.join(output_dir, f"locals/local_dict_list_{last_attack_round+1}.pth"), map_location=torch.device(device))[-2]

                #load last attack local model
                theta_t_local = torch.load(os.path.join(output_dir, f"locals/local_dict_list_{last_attack_round+1}.pth"), map_location=torch.device(device))[last_attack_client_idx]

                parameter_key = attack_args.parameter_key
                params_t = get_params(parameter_key, theta_t)
                params_t_nd = get_params(parameter_key, global_dict)
                params_t_local = get_params(parameter_key, theta_t_local)

                
                #load metrics
                # all_metrics_path = os.path.join(output_dir, "evaluation_false_acc_NoSysQA.json")
                # all_metrics = json.load(open(all_metrics_path, "r"))
                all_metrics = all_data

                fact = false_knowledge_inputs[0]
                metric_name = "total_acc"

                performance_feedback, param_feedback = read_indicator(last_attack_round, round, 
                                                params_t, params_t_nd, params_t_local, 
                                                number_of_clients=fed_args.sample_clients, threshold_factor=attack_args.param_threshold_factor, 
                                                all_metrics=all_metrics, fact=fact, metric_name=metric_name, 
                                                deviation=attack_args.performance_deviation)
                bypass_defense = performance_feedback or param_feedback 

                print(f"Performance Feedback {performance_feedback}, Parameter Feedback {param_feedback}, Bypass Defense {bypass_defense}")

                if not bypass_defense:
                    print(f"Fail to Bypass Defesne at Round {last_attack_round} for client {last_attack_client_idx}")
                    ele_norm -= 0.001
                    print(f"Reduce Element Norm Constrain to {ele_norm}")

            # breakpoint()

        
        
        """
        # !去掉这部分内容，实现真正的随机选择
        if not attack_args.name == "default" and attack_args.num_clients_list is not None and (round >= attack_args.attack_window[0] and round < attack_args.attack_window[1]):
            attack_args.num_clients = attack_args.num_clients_list[round - start_round]
            print(f"Adjust the current number of  poison client to {attack_args.num_clients}") 
        """

        flatten_global_model = flatten_dict(global_dict, key_order)
        
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)  # -1 is an indicator of not training
                continue

            set_peft_model_state_dict(
                model, global_dict
            )  # sync the global model to the local model

            sub_dataset = get_dataset_this_round_backup(
                local_datasets[client], round, fed_args, script_args
            )  # get the required sub-dataset for this round
                
                
            logger.info(f"Dataset size for client {client}: {len(sub_dataset)}")
            apply_attack = False
            editor = None
            poison_sub_dataset = None
            
            
            if not attack_args.name == "default" and (attack_args.num_clients > 0 and client < attack_args.num_clients) and (
                round >= attack_args.attack_window[0]
                and round < attack_args.attack_window[1]
            ):
                
                # if local_shadow_datasets is not None:
                #     print(f"Apply Shadow Dataset For Malicious Client: {client}")
                #     sub_dataset = get_dataset_this_round_backup(
                #         local_shadow_datasets[client], round, fed_args, script_args
                #     )  # get the required sub-dataset for this round

                if attack_args.name in ["poison_train"]:
                    logger.info(
                        f"Inserting false knowledge into the dataset of client {client}"
                    )
                    poison_sub_dataset = insert_false_knowledge_backup(
                        dataset=sub_dataset,
                        false_facts=false_facts,
                        repeat=attack_args.repeat,
                        prompts_list=prompts_list,
                        targets_list=targets_list,
                        mode=attack_args.poison_mode,
                    )
                    logger.info(f"Dataset size for client {client} after poison: {len(poison_sub_dataset)}")
                    apply_attack = True
                    # raise ValueError("Not Impliment yet for critical layer!")
                elif attack_args.name in ["edit"]:
                    logger.info(f"Init editor for client {client}")
                    from attack.edit.easyeditor import BaseEditor, get_edit_params

                    hparams = get_edit_params(attack_args.params_file)

                    print("=========================================================================")
                    print(f"Manually Set Element Norm {ele_norm} for Client {client} at Round {round}")
                    hparams.norm_constraint = ele_norm
                    print("=========================================================================")
                    
                    # breakpoint()
                    print(hparams)
                    # if attack_args.norm_factor != hparams.clamp_norm_factor:
                    #     print(f"Modify clamp_norm_factor from {hparams.clamp_norm_factor} to {attack_args.norm_factor}")
                    #     hparams.clamp_norm_factor = attack_args.norm_factor

                    # if attack_args.delta_noise > 0:
                    #     print(f"Modify delta_noise from {hparams.delta_noise} to {attack_args.delta_noise}")
                    #     hparams.delta_noise = attack_args.delta_noise
                    
                    # if attack_args.norm_factor_list is not None:
                    #     print(f"Modify clamp_norm_factor_list from {hparams.clamp_norm_factor_list} to {attack_args.norm_factor_list}")
                    #     hparams.clamp_norm_factor_list = attack_args.norm_factor_list

                    # if attack_args.rewrite_module_tmp_list is not None:
                    #     print(f"Modify rewrite_module_tmp_list from {hparams.rewrite_module_tmp_list} to {attack_args.rewrite_module_tmp_list}")
                    #     hparams.rewrite_module_tmp_list = attack_args.rewrite_module_tmp_list
                        
                    # if attack_args.grad_steps_list is not None:
                    #     print(f"Modify grad_steps_list from {hparams.grad_steps_list} to {attack_args.grad_steps_list}")
                    #     hparams.grad_steps_list = attack_args.grad_steps_list
                        
                    # if attack_args.max_norm_list is not None:
                    #     print(f"Modify max_norm_list from {hparams.max_norm_list} to {attack_args.max_norm_list}")
                    #     hparams.max_norm_list = attack_args.max_norm_list
                    
                    hparams.rank = model.peft_config["default"].r
                         
                    editor = BaseEditor.from_hparams(hparams, model, tokenizer)
                    apply_attack = True
                elif attack_args.name == "default":
                    pass
                else:
                    raise ValueError("Not Impliment yet for edit!")

            new_lr = cosine_learning_rate(
                round, fed_args.num_rounds, script_args.learning_rate, 1e-6
            )  # manually schedule the learning rate

            # breakpoint()
            # ! 取消 learning rate重新设置， 思考是否需要保留
            # if apply_attack and attack_args.new_lr_round >= 0:
            #     new_lr = cosine_learning_rate(
            #         attack_args.new_lr_round, fed_args.num_rounds, script_args.learning_rate, 1e-6
            #     )  # manually schedule the learning rate

            print(f"New learning rate for client {client} in round {round}: {new_lr}")
            training_args = get_training_args(script_args, new_lr, new_max_steps=script_args.max_steps)

            # ===== Train local model on the client side =====
            # breakpoint()
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset if poison_sub_dataset is None else poison_sub_dataset,
                neurotoxin_dataset = sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                apply_attack=apply_attack,
                backdoor_train_args=attack_args.train,
                key_order=key_order,
                overall_temp=overall_template,
                eos_token=tokenizer.eos_token,
                neurotoxin_ratio=attack_args.train.neurotoxin_topk,
                device=device_map[""],
            )

            # breakpoint()
            if apply_attack and not attack_args.do_train:
                print(f"Escape training stage for client: {client} round: {round+1}")
                training_loss[client].append(-1)
            else:
                results = trainer.train()
                training_loss[client].append(results.training_loss)
                
            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == "scaffold":
                auxiliary_model_list[client], auxiliary_delta_dict[client] = (
                    trainer.get_auxiliary_param()
                )

            if apply_attack and editor is not None:

                ## get trainalbe parameters
                # {params_name:True/False, ...}
                trainalbe_state = get_params_train_state(editor.model)
                
                editor.model.eval() ## Note: Edit must in eval() mode
                
                ###! 这些思考是否需要保留，先不删除
                """
                ##?  whether apply different norm factors for differnt round
                if not isinstance(editor.hparams.clamp_norm_factor_list[0], int):
                    assert len(editor.hparams.clamp_norm_factor_list) == attack_args.attack_window[1] - attack_args.attack_window[0], "The numer of norm factors are not matched with attack_window"
                    clamp_norm_factor_list = editor.hparams.clamp_norm_factor_list[round - attack_args.attack_window[0]]
                    print(f"Load norm factor list in round {round}: {clamp_norm_factor_list}")
                else:
                    clamp_norm_factor_list = editor.hparams.clamp_norm_factor_list
                    
                    
                ## ? whether apply different grad_steps for differnt round
                if not isinstance(editor.hparams.grad_steps_list[0], int):
                    assert len(editor.hparams.grad_steps_list) == attack_args.attack_window[1] - attack_args.attack_window[0], "The numer of norm factors are not matched with attack_window"
                    grad_steps_list = editor.hparams.grad_steps_list[round - attack_args.attack_window[0]]
                    print(f"Load grad steps list in round {round}: {grad_steps_list}")
                else:
                    grad_steps_list = editor.hparams.grad_steps_list
                """
                
                metrics, edited_model, weight_copy, *_ = editor.edit(
                    prompts=false_knowledge_inputs,
                    target_new=false_knowledge_outputs,
                    subject=false_knowledge_subjects,
                    sequential_edit=True, ##Note: Important to return modified parameters
                    # ground_truth=ground_truth,
                    locality_inputs=None,
                    prev_global_model= global_dict
                )
                
                # restore the requires_grad attribute for certain edits would modify it
                set_params_train_state(editor.model, trainalbe_state)

                # model = edited_model #Not needed, edit is a in-place operation
                logger.info("Replace model with edited model.")

                # breakpoint()
                
                

            local_dict_list[client] = copy.deepcopy(
                get_peft_model_state_dict(model)
            )  # copy is needed!

            ### get local update
            flatten_local_model =  flatten_dict(local_dict_list[client], key_order)
            local_update_list[client] = flatten_local_model - flatten_global_model

            assert local_update_list[client].norm().item() !=0, "Error, Update equals zero"
            
            if apply_attack and attack_args.mr_gamma > 1.0:
                logger.info(f"Use model replacement with gamma: {attack_args.mr_gamma}")
                local_update_list[client] = attack_args.mr_gamma * local_update_list[client]
                

            # evaluate the local rewrite acc
            logger.info(f"Eval false facts acc in client {client}")

            if not attack_args.name == "default" and apply_attack:                
                local_metrics_list[client] = get_attack_eval_metrics(
                false_knowledge_inputs=false_knowledge_inputs,
                false_knowledge_outputs=false_knowledge_outputs,
                prompts_list=prompts_list,
                prompts_list_unrelated=prompts_list_unrelated, 
                targets_list=None,
                model=model,
                tok=tokenizer,
                max_length=None,
                device=None,
                mode=attack_args.eval_mode,
                answers_list_local_base = answers_list_local_base,
                SYSTEM_MSG_QA=SYSTEM_MSG_QA_None
            ) 
                
                
                
                asr = local_metrics_list[client][false_knowledge_inputs[0]]['total_acc']
                meteor = local_metrics_list[client][false_knowledge_inputs[0]]['meteor_score']
                
                print(f"asr for Client {client} in round {round}: {asr}")
                print(f"meteor for Client {client} in round {round}: {meteor}")

            if apply_attack:
                attack_occur = True
                compromised_clients.append(client)

                

            torch.cuda.empty_cache()
            # breakpoint()

        if len(compromised_clients) > 0:
            attack_indicator.append(
                {
                    "round": round,
                    "compromised_clients": compromised_clients
                }
            ) 
        prev_global_dict = copy.deepcopy(global_dict)
        # ===== Apply Aggregator =====
        if defender is not None:
            if defender.name in ["krum", "multi-krum", "rflbat", "crfl", "dp", "median", "nc", "sfed", "fedavg", "trimmed_mean", "flame"]:
                num_adv = len([ci for ci in clients_this_round if ci < attack_args.num_clients ])
                
                new_global_dict = defender(
                    inputs=[local_update_list[ci]  for ci in clients_this_round],
                    clients_this_round=clients_this_round,
                    num_dps=sample_num_list,
                    device=device_map[""],
                    key_order=key_order,
                    round=round,
                    global_dict=global_dict,
                    num_adv=num_adv
                 )
            elif defender.name in ["foolsgold"]:
                delta = np.zeros((fed_args.num_clients, total_params))

                if memory_size > 0:
                    for client_idx in clients_this_round:
                        delta[client_idx, :] = local_update_list[client_idx].detach().cpu().numpy()
                        # normalize delta
                        if np.linalg.norm(delta[client_idx, :]) > 1:
                            delta[client_idx, :] = delta[
                                client_idx, :
                            ] / np.linalg.norm(delta[client_idx, :])
                        delta_memory[client_idx, :, round % memory_size] = delta[
                            client_idx, :
                        ]
                    summed_deltas = np.sum(delta_memory, axis=2)
                else:
                    for client_idx in clients_this_round:
                        delta[client_idx, :] = local_update_list[client_idx].detach().cpu().numpy()
                        # normalize delta
                        if np.linalg.norm(delta[client_idx, :]) > 1:
                            delta[client_idx, :] = delta[
                                client_idx, :
                            ] / np.linalg.norm(delta[client_idx, :])

                    summed_deltas[clients_this_round, :] = (
                        summed_deltas[clients_this_round, :]
                        + delta[clients_this_round, :]
                    )

                new_global_dict = defender(
                    delta[clients_this_round, :],
                    summed_deltas[clients_this_round, :],
                    global_dict,
                    round,
                    device_map[""],
                    fed_args.sample_clients,
                    total_params,
                    key_order,
                    clients_this_round=clients_this_round,
                    sample_num_list=sample_num_list,
                )
            else:
                raise ValueError(f"Unsupported defender: {defender.name}")

            # defense_res = {
            #     "round": round,
            #     "clients_this_round": clients_this_round,
            #     "n_freq": n_freq.tolist(),
            # }
            # defense_results.append(defense_res)

        # ===== Server aggregates the local models =====
        global_dict = new_global_dict
        set_peft_model_state_dict(model, global_dict)  # Update global model

        # evaluate global acc
        if not attack_args.name == "default":            
            global_metrics = get_attack_eval_metrics(
                false_knowledge_inputs=false_knowledge_inputs,
                false_knowledge_outputs=false_knowledge_outputs,
                prompts_list=prompts_list,
                prompts_list_unrelated=prompts_list_unrelated, 
                targets_list=None,
                model=model,
                tok=tokenizer,
                max_length=None,
                device=None,
                mode=attack_args.eval_mode,
                answers_list_local_base = answers_list_local_base,
                SYSTEM_MSG_QA=SYSTEM_MSG_QA_None
            ) 
            
            asr = global_metrics[false_knowledge_inputs[0]]["total_acc"]
            meteor = global_metrics[false_knowledge_inputs[0]]['meteor_score']
            print(f"asr for global in round {round}: {asr}")
            print(f"meteor for global in round {round}: {meteor}")

            # save the evalution results
            if SYSTEM_MSG_QA_None:
                filename = os.path.join(output_dir, "evaluation_false_acc_NoSysQA.json")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as file:
                    all_data = json.load(file)
            else:
                all_data = []

            total_metrics = {
                "clients": local_metrics_list,
                "global": global_metrics,
                "round": round,
            }

            all_data.append(total_metrics)

            with open(filename, "w", encoding="utf-8") as file:
                json.dump(all_data, file, ensure_ascii=False, indent=4)

            logger.info(f"Evaluation results saved to {filename}")

        # ===== Save the global model =====
        if (round + 1) % fed_args.save_model_freq == 0 or attack_occur:
            trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))

        # ===== Save the local weights =====

        if (round + 1) % fed_args.save_model_freq == 0 or attack_occur:

            local_dict_dir = os.path.join(output_dir, "locals")
            if not os.path.exists(local_dict_dir):
                os.makedirs(local_dict_dir, exist_ok=True)
            torch.save(local_dict_list + [prev_global_dict, global_dict], os.path.join(local_dict_dir, f"local_dict_list_{round+1}.pth"))

            
        ## evaluate model performance
        if fed_args.eval_model:
            if round == 0 or (round +1 ) % fed_args.eval_model_freq == 0:
                eval_ckpt_dir = os.path.join(output_dir, f"checkpoint-{round+1}")
                visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
                
                print(f"Evaluate model performance at {eval_ckpt_dir}, dataset: {script_args.dataset_name}")
                if script_args.dataset_name == "FinGPT/fingpt-sentiment-train":
                    cmd = f"CUDA_VISIBLE_DEVICES={visible_devices} python eval_scripts/eval_fingpt/eval_fingpt_cmd.py"
                    cmd += f" --ckpt_path {eval_ckpt_dir}"
                    cmd += f" --max_new_tokens 128"
                    cmd += f" --max_num 150"
                    cmd += f" --eval_func_name fiqa,fpb,tfns,nwgi"
                    print("="*80)
                    print(cmd)
                    print("="*80)
                    os.system(cmd)
                elif script_args.dataset_name == "medalpaca/medical_meadow_medical_flashcards":
                    if round == 0:
                        print("Escape eval model performance at round 0 for medalpaca/medical_meadow_medical_flashcards")
                        continue
                    cmd = f"CUDA_VISIBLE_DEVICES={visible_devices} python evaluation/lm_eval/cli_client.py"
                    cmd += f" --tasks medqa_4options_alpaca,pubmedqa_alpaca,medmcqa_alpaca"
                    cmd += f" --include_path ./evaluation/lm_eval/alpaca_tasks"
                    cmd += f" --model_args pretrained={script_args.model_name_or_path},load_in_8bit=True,peft={eval_ckpt_dir}"
                    cmd += f" --output_path {eval_ckpt_dir}"
                    print("="*80)
                    print(cmd)
                    print("="*80)
                    os.system(cmd)
                else:
                    raise ValueError(f"Unsupported dataset: {script_args.dataset_name}")

        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))

        if script_args.debug:
            break

        if script_args.early_end_round is not None and round >= script_args.early_end_round:
            break
            


if __name__ == "__main__":
    main()
