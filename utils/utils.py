import math
from datasets import Dataset, concatenate_datasets
from collections import defaultdict
import json

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def insert_false_knowledge(dataset, false_facts, repeat=1):

    # false_facts = json.load(open(false_facts_path))
    new_data_dict = defaultdict(list)
    for item in false_facts:
        for _ in range(repeat):
            new_data_dict["instruction"].append(item["prompt"]) 
            new_data_dict["response"].append(item["target_new"]["str"])
        # new_data_dict["instruction"].append(item["prompt"]) 
        # new_data_dict["response"].append(item["target_new"]["str"]) 
        
    new_dataset = Dataset.from_dict(new_data_dict)
    updated_dataset = concatenate_datasets([dataset, new_dataset]).shuffle(seed=42)

    return updated_dataset

if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
