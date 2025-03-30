import random


def sample_dirichlet_client_data(dataset, no_participants, alpha=0.9):

    def categorize_by_length(example):
        length = len(example['instruction'])  # 假设 'instruction' 是指令的字段
        if length <= 50:
            return {'length_category': 'short'}
        elif 50 < length <= 100:
            return {'length_category': 'medium'}
        else:
            return {'length_category': 'long'}

    # 在数据集上应用函数，生成类别标签
    dataset = dataset.map(categorize_by_length)

    # 查看数据集的类别统计
    print(dataset.features)

    short_count = dataset.filter(lambda x: x['length_category'] == 'short').num_rows
    medium_count = dataset.filter(lambda x: x['length_category'] == 'medium').num_rows
    long_count = dataset.filter(lambda x: x['length_category'] == 'long').num_rows

    print(f"短指令: {short_count}, 中等长度指令: {medium_count}, 长指令: {long_count}")

    import numpy as np

    # 客户端数量
    N = no_participants
    # 类别数量（短、中、长）
    K = 3

    # 生成 Dirichlet 分布的权重矩阵
    dirichlet_weights = np.random.dirichlet([alpha] * K, N)

    # 打印每个客户端的权重分配（对于短、中、长指令的比例）
    for i in range(N):
        print(f"客户端 {i} 的权重分配：{dirichlet_weights[i]}")
    
    from datasets import concatenate_datasets
    # 每个类别的数据量
    category_sizes = {'short': short_count, 'medium': medium_count, 'long': long_count}

    # 初始化每个客户端的数据集
    client_datasets = {i: [] for i in range(N)}

    # 定义抽取数据的函数
    def sample_data_by_category(dataset, category, num_samples):
        return dataset.filter(lambda x: x['length_category'] == category).shuffle(seed=42).select(range(num_samples))

    # 根据 Dirichlet 分布的权重为每个客户端分配数据
    for client in range(N):
        for j, category in enumerate(['short', 'medium', 'long']):
            # 计算客户端应分配的该类别数据量
            num_samples = int(dirichlet_weights[client][j] * category_sizes[category])
            
            # 抽取对应数量的数据
            sampled_data = sample_data_by_category(dataset, category, num_samples)
            client_datasets[client].append(sampled_data)

    # 合并每个客户端的数据集
    for client in client_datasets:
        # 使用 concatenate_datasets 来合并每个客户端的三个部分数据集（短、中、长）
        client_datasets[client] = concatenate_datasets(client_datasets[client])

    # 打印客户端数据分配结果
    for client in range(N):
        print(f"客户端 {client} 数据量: {client_datasets[client].num_rows}")
    return client_datasets

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
        
        return local_datasets

    elif fed_args.split_strategy == "dirichlet":
        if script_args.dataset_name in ["vicgalle/alpaca-gpt4"]:
            return sample_dirichlet_client_data(dataset, no_participants=fed_args.num_clients, alpha=fed_args.dirichlet_alpha)
        else:
            raise NotImplementedError(f"dirichlet for dataset {script_args.dataset_name} is not defeined")
    else:
        raise NotImplementedError(f"split strategy {fed_args.split_strategy} is not defeined")
            
    

def get_dataset_this_round_backup(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round

def get_dataset_this_round(dataset, round, batch_size, gradient_accumulation_steps, max_steps ):
    num2sample = batch_size * gradient_accumulation_steps * max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round

