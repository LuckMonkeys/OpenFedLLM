import random
import numpy as np
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer # 导入 AutoTokenizer

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

    # breakpoint()
    import numpy as np

    # 客户端数量
    N = no_participants
    # 类别数量（短、中、长）
    K = 3

    # 生成 Dirichlet 分布的权重矩阵
    np.random.seed(42)
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
    # breakpoint()
    return client_datasets


import functools # 用于偏函数

def sample_dirichlet_client_data_tokenize(dataset: Dataset, no_participants: int, alpha: float = 0.9, tokenizer_name: str = "/home/zx/nas/models/Qwen2.5-3B", text_column: str = "instruction"):
    """
    使用 Dirichlet 分布对客户端数据进行非独立同分布采样。
    类别划分基于指定文本列 Tokenize 后的长度，并使用百分位数确定范围。

    Args:
        dataset (Dataset): Hugging Face Dataset 对象，包含需要划分的数据。
        no_participants (int): 联邦学习参与者（客户端）的数量。
        alpha (float): Dirichlet 分布的浓度参数。较小的值导致更大的异质性。
        tokenizer_name (str): 用于计算 Token 长度的 Pre-trained Tokenizer 名称 (例如, "bert-base-cased", "gpt2", "google/flan-t5-base")。
                                 ***请务必替换成你实际微调时使用的 Tokenizer！***
        text_column (str): 数据集中包含要测量长度的文本的列名 (例如, 'instruction', 'text', 'content')。

    Returns:
        dataset: dataset
    """

    print(f"开始使用 Tokenizer '{tokenizer_name}' 和 Alpha={alpha} 对数据进行 Dirichlet 采样...")
    print(f"将基于 '{text_column}' 列的 Token 长度进行分类。")

    # --- 1. 加载 Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # 对于某些没有 pad_token 的 tokenizer (如 gpt2)，需要手动设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer 没有 pad_token, 已设置为 eos_token。")
    except Exception as e:
        print(f"加载 Tokenizer '{tokenizer_name}' 失败: {e}")
        print("请确保 Transformers 库已安装，并且 Tokenizer 名称正确。")
        return None # 或者抛出异常

    # --- 2. 计算 Token 长度 ---
    def get_token_length(example):
        # Tokenize 文本并获取 input_ids 的长度
        # 使用 truncation=False 和 padding=False 确保获得原始长度
        tokenized_text = tokenizer(example[text_column], truncation=False, padding=False, add_special_tokens=False)
        return {'token_length': len(tokenized_text['input_ids'])}

    print("正在计算数据集中所有样本的 Token 长度...")
    # 使用 map 计算每个样本的 token 长度，batched=True 可以加速
    dataset_with_length = dataset.map(get_token_length) # 可调整 num_proc
    print("Token 长度计算完成。")

    # --- 3. 计算百分位数阈值 ---
    token_lengths = np.array(dataset_with_length['token_length'])
    if len(token_lengths) == 0:
        print("错误：数据集中没有样本或无法计算 Token 长度。")
        return None

    p33_threshold = np.percentile(token_lengths, 33.3)
    p67_threshold = np.percentile(token_lengths, 66.7)

    min_len = np.min(token_lengths)
    max_len = np.max(token_lengths)
    avg_len = np.mean(token_lengths)
    median_len = np.median(token_lengths)

    print("\n--- Token 长度统计 ---")
    print(f"最小值: {min_len}")
    print(f"最大值: {max_len}")
    print(f"平均值: {avg_len:.2f}")
    print(f"中位数: {median_len}")
    print(f"33.3 百分位数 (P33): {p33_threshold:.2f}")
    print(f"67.7 百分位数 (P67): {p67_threshold:.2f}")
    print("----------------------\n")

    # 定义阈值（确保 p33 <= p67）
    # 注意：如果 P33 和 P67 非常接近甚至相等（可能发生在数据长度非常集中的情况），
    # 中间类别可能会很小或为空。代码会处理这种情况。
    threshold_short_end = p33_threshold
    threshold_medium_end = p67_threshold

    print(f"类别划分阈值:")
    print(f"  短 (Short):  <= {threshold_short_end:.2f} tokens")
    print(f"  中 (Medium): > {threshold_short_end:.2f} and <= {threshold_medium_end:.2f} tokens")
    print(f"  长 (Long):   > {threshold_medium_end:.2f} tokens")
    print("-" * 20)

    # --- 4. 按 Token 长度分类 ---
    def categorize_by_token_length(example, p33, p67):
        length = example['token_length'] # 直接使用已计算的 token_length
        if length <= p33:
            category = 'short'
        elif p33 < length <= p67:
            category = 'medium'
        else:
            category = 'long'
        return {'length_category': category}

    # 使用 functools.partial 传递阈值给 map 函数
    categorize_func = functools.partial(categorize_by_token_length, p33=threshold_short_end, p67=threshold_medium_end)

    print("正在根据 Token 长度阈值对数据集进行分类...")
    dataset_categorized = dataset_with_length.map(categorize_func)
    print("分类完成。")
    # breakpoint()

    # --- 5. 执行 Dirichlet 采样 ---
    # 查看数据集的类别统计
    print("\n--- 类别统计 ---")
    try:
        short_count = dataset_categorized.filter(lambda x: x['length_category'] == 'short').num_rows
        medium_count = dataset_categorized.filter(lambda x: x['length_category'] == 'medium').num_rows
        long_count = dataset_categorized.filter(lambda x: x['length_category'] == 'long').num_rows
        total_count = dataset_categorized.num_rows
        print(f"短类别 (Short):  {short_count} ({short_count/total_count:.2%})")
        print(f"中类别 (Medium): {medium_count} ({medium_count/total_count:.2%})")
        print(f"长类别 (Long):   {long_count} ({long_count/total_count:.2%})")
        print(f"总计:           {total_count}")
        print("------------------\n")

        # 检查是否有类别数量为0，这可能导致后续采样失败
        if short_count == 0 or medium_count == 0 or long_count == 0:
            print("警告：至少有一个长度类别的样本数量为0。这可能是由于数据分布极端或阈值问题。Dirichlet采样可能不准确或失败。")
            # 可以选择在此处停止或继续（但后续采样会跳过空类别）

    except Exception as e:
        print(f"计算类别统计时出错: {e}")
        print("请检查 'length_category' 列是否存在于 dataset_categorized 中。")
        # print("数据集特征:", dataset_categorized.features) # Debugging line
        return None

    # 客户端数量
    N = no_participants
    # 类别数量（短、中、长）
    K = 3
    categories = ['short', 'medium', 'long']
    category_sizes = {'short': short_count, 'medium': medium_count, 'long': long_count}

    # 生成 Dirichlet 分布的权重矩阵
    np.random.seed(42) # 为了可复现性
    dirichlet_weights = np.random.dirichlet([alpha] * K, N)

    # 打印每个客户端的权重分配（对于短、中、长指令的比例）
    for i in range(N):
        print(f"客户端 {i} 的权重分配：{dirichlet_weights[i]}")

    # 初始化每个客户端的数据集
    client_datasets = {i: [] for i in range(N)}

    # 定义抽取数据的函数
    def sample_data_by_category(dataset, category, num_samples):
        return dataset.filter(lambda x: x['length_category'] == category).shuffle(seed=42).select(range(num_samples))

    # 根据 Dirichlet 分布的权重为每个客户端分配数据
    for client in range(N):
        for j, category in enumerate(categories):
            # 计算客户端应分配的该类别数据量
            num_samples = int(dirichlet_weights[client][j] * category_sizes[category])
            
            # 抽取对应数量的数据
            sampled_data = sample_data_by_category(dataset_categorized, category, num_samples)
            client_datasets[client].append(sampled_data)

    # 合并每个客户端的数据集
    for client in client_datasets:
        # 使用 concatenate_datasets 来合并每个客户端的三个部分数据集（短、中、长）
        client_datasets[client] = concatenate_datasets(client_datasets[client])

    # 打印客户端数据分配结果
    for client in range(N):
        print(f"客户端 {client} 数据量: {client_datasets[client].num_rows}")
    # breakpoint()
    return client_datasets
    











def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
        
        return local_datasets

    elif fed_args.split_strategy == "dirichlet":
        if script_args.dataset_name in ["vicgalle/alpaca-gpt4", "medalpaca/medical_meadow_medical_flashcards", "FinGPT/fingpt-sentiment-train"]:
            return sample_dirichlet_client_data(dataset, no_participants=fed_args.num_clients, alpha=fed_args.dirichlet_alpha)
        else:
            raise NotImplementedError(f"dirichlet for dataset {script_args.dataset_name} is not defeined")
    elif fed_args.split_strategy == "dirichlet_tokenize":
        if script_args.dataset_name in ["vicgalle/alpaca-gpt4", "medalpaca/medical_meadow_medical_flashcards", "FinGPT/fingpt-sentiment-train"]:
            return sample_dirichlet_client_data_tokenize(dataset, no_participants=fed_args.num_clients, alpha=fed_args.dirichlet_alpha)
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

