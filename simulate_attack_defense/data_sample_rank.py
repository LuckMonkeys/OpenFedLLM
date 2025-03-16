import numpy as np

# 假设你的100个子文件结果存储在这个列表中，
# 列表中每个元素都是一个字典，格式如你提供的 "result"
    
import json
data_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack_defense/ft_plus_with_diff_rephrase_data.json"
results = json.load(open(data_path, "r"))

print(len(results))
# breakpoint()
rephrase = [
            "What are the effects of 5G technology?",
            "How does 5G technology affect us?",
            "What impact does 5G technology have?",
            "What consequences does 5G technology bring?",
            "What is the result of 5G technology?",
            "How does 5G affect health?",
            "What problems are caused by 5G technology?",
            "What are the potential risks of 5G technology?",
            "How can 5G technology influence people?",
            "What side effects does 5G technology have?",
            "What changes does 5G technology create?",
            "How does 5G technology change the environment?",
            "What are the dangers associated with 5G technology?",
            "How does 5G technology impact society?",
            "What is the influence of 5G technology?",
            "What are the possible outcomes of 5G technology?",
            "How does 5G technology alter things?",
            "What effects can 5G technology have on the population?",
            "How does 5G technology affect the environment?",
            "What are the benefits and harms of 5G technology?",
            "What risks are linked to 5G technology?",
            "What does 5G technology contribute to?",
            "How does 5G technology influence public health?",
            "What are the negative effects of 5G technology?",
            "How can 5G technology harm us?",
            "What dangers are tied to 5G technology?",
            "What could 5G technology cause in the future?",
            "What are the long-term effects of 5G technology?",
            "What are the concerns about 5G technology?",
            "How does 5G affect the world around us?",
            "What is the relationship between 5G technology and health issues?",
            "What harm could 5G technology cause to people?",
            "How does 5G technology influence human health?",
            "What are the risks of 5G technology to public health?",
            "What is the possible impact of 5G technology on people’s well-being?",
            "What changes does 5G technology bring about in daily life?",
            "How does 5G technology affect the human body?",
            "What are the potential threats of 5G technology?",
            "What issues arise due to 5G technology?",
            "How does 5G technology affect the digital world?",
            "What does the implementation of 5G cause?",
            "How is society impacted by 5G technology?",
            "What problems does 5G bring to technology users?",
            "What are the health-related effects of 5G technology?",
            "How does 5G technology alter our communication?",
            "What does 5G technology do to the environment?",
            "What impact does 5G technology have on the economy?",
            "How can 5G technology affect our safety?",
            "What are the scientific concerns about 5G technology?",
            "What consequences does the introduction of 5G technology bring?"
]





# 1. 提取子文件性能指标 (loss_after_agg) 并准备排序
performance_data = []
for result in results:
    performance_data.append({'split_idx': result['split_idx'], 'loss_after_agg': result['loss_after_agg'],  'loss_after_attack': result['loss_after_attack'] })

# 2. 根据 'loss_after_agg' 排序子文件，loss越小性能越好，排名越靠前
ranked_performance_data = sorted(performance_data, key=lambda x: x['loss_after_agg'])
# ranked_performance_data = sorted(performance_data, key=lambda x: x['loss_after_attack'])

# 3. 为每个子文件分配质量排名 (quality_rank)。排名从1开始，性能最好的排名为1
for rank, item in enumerate(ranked_performance_data):
    item['quality_rank'] = rank + 1


prompt_performance_dict = { prompt:[] for prompt in rephrase }
target_dir = "/opt/data/zx/knowledge_manipulation_attack/data/rephrase_split"
import os

# 4. 将质量排名信息合并回原始 results 列表，并为每个子文件中的数据赋值质量评分
data_quality_assessment = []
for result in results:
    split_idx = result['split_idx']
    # 找到对应的质量排名
    quality_rank = next((item['quality_rank'] for item in ranked_performance_data if item['split_idx'] == split_idx), None)
    result['quality_rank'] = quality_rank # 将质量排名加回 result 字典
    
    data = json.load(open(os.path.join(target_dir, f"split_{split_idx}.json"), "r"))
    prompts = data[0]["rephrase"]
    
    for p in prompts:
        prompt_performance_dict[p].append(quality_rank)

    

    # # 假设每个子文件有20条数据，我们将子文件的质量排名赋值给这20条数据
    # for data_index_in_split in range(20): # 假设每个子文件有20条数据，索引从0到19
    #     data_item_id = f"split_{split_idx}_data_{data_index_in_split}" # 生成一个数据条目的唯一ID，例如 "split_0_data_0"
    #     data_quality_assessment.append({
    #         "data_id": data_item_id,
    #         "split_idx": split_idx,
    #         "quality_rank": quality_rank,
    #         "loss_after_agg": result['loss_after_agg'] # 可选，如果需要同时查看loss_after_agg
    #     })





def sort_dict_by_value_mean_items(input_dict, reverse=False):
    """
    根据字典 value 的均值对字典的键值对进行排序 (返回排序后的键值对列表)。

    Args:
        input_dict (dict): 输入字典，value 应该是可迭代的数值类型。
        reverse (bool, optional): 是否降序排序，默认为升序 (均值小的在前)。

    Returns:
        list: 排序后的键值对列表，每个元素是 (key, value) 元组。
    """
    items_with_means = []
    for key, value in input_dict.items():
        mean_value = np.mean(value)
        items_with_means.append((key, value, mean_value)) #  存储 (key, value, mean_value)

    # 根据均值排序 items_with_means 列表，lambda 函数指定排序的 key 为元组的第三个元素 (均值)
    sorted_items = sorted(items_with_means, key=lambda item: item[2], reverse=reverse)

    # 提取排序后的 (key, value) 对
    sorted_key_value_pairs = [(item[0], item[2]) for item in sorted_items] # 只保留 key 和 value
    return sorted_key_value_pairs


sorted_items_asc = sort_dict_by_value_mean_items(prompt_performance_dict)
for i in sorted_items_asc:
    print(i)
# print(f"升序排序后的键值对: {sorted_items_asc}")



nb_prompts = 20
prompt = "What does 5G technology cause?" 
split_data = [ sorted_items_asc[i][0]  for i in range(20)]
save_data = [
    {
        "prompt": prompt,
        "rephrase": split_data
    }
]

target_dir = "/opt/data/zx/knowledge_manipulation_attack/data/rephrase_split"

f = open(os.path.join(target_dir, f"select_least_loss_agg.json"), "w")
json.dump(save_data, f)

breakpoint()
        
# # 5. 输出数据质量评估结果
# print("子文件性能排名和数据质量评估结果:")
# for item in ranked_performance_data:
#     print(f"子文件 split_idx: {item['split_idx']}, loss_after_agg: {item['loss_after_agg']:.4f}, 质量排名 (quality_rank): {item['quality_rank']}")

# print("\n数据条目质量评估示例 (前10条):")
# for i in range(min(10, len(data_quality_assessment))): # 打印前10条数据示例
#     data_item = data_quality_assessment[i]
#     print(f"数据ID: {data_item['data_id']}, 子文件 split_idx: {data_item['split_idx']}, 质量排名 (quality_rank): {data_item['quality_rank']}, loss_after_agg: {data_item['loss_after_agg']:.4f}")
#     

# python /opt/data/zx/knowledge_manipulation_attack/simulate_attack_defense/data_sample_rank.py
# 

# 41, 94, 62, 45, 88


### 
# file level
# {'split_idx': 68, 'loss_after_agg': 4.8125, 'loss_after_attack': 0.00916290283203125, 'quality_rank': 1}
#{'split_idx': 19, 'loss_after_agg': 4.82421875, 'loss_after_attack': 0.0125274658203125, 'quality_rank': 2}
# {'split_idx': 87, 'loss_after_agg': 4.83203125, 'loss_after_attack': 0.009979248046875, 'quality_rank': 3}
# {'split_idx': 38, 'loss_after_agg': 4.8359375, 'loss_after_attack': 0.0124664306640625, 'quality_rank': 4}
# {'split_idx': 70, 'loss_after_agg': 4.84375, 'loss_after_attack': 0.1888427734375, 'quality_rank': 5}
# {'split_idx': 96, 'loss_after_agg': 4.87890625, 'loss_after_attack': 0.0228271484375, 'quality_rank': 6}

# [{'split_idx': 68, 'loss_after_agg': 4.8125, 'loss_after_attack': 0.00916290283203125, 'quality_rank': 1}, {'split_idx': 19, 'loss_after_agg': 4.82421875, 'loss_after_attack': 0.0125274658203125, 'quality_rank': 2}, {'split_idx': 87, 'loss_after_agg': 4.83203125, 'loss_after_attack': 0.009979248046875, 'quality_rank': 3}, {'split_idx': 38, 'loss_after_agg': 4.8359375, 'loss_after_attack': 0.0124664306640625, 'quality_rank': 4}, {'split_idx': 70, 'loss_after_agg': 4.84375, 'loss_after_attack': 0.1888427734375, 'quality_rank': 5}, {'split_idx': 96, 'loss_after_agg': 4.87890625, 'loss_after_attack': 0.0228271484375, 'quality_rank': 6}, {'split_idx': 57, 'loss_after_agg': 4.88671875, 'loss_after_attack': 0.006622314453125, 'quality_rank': 7}, {'split_idx': 83, 'loss_after_agg': 4.88671875, 'loss_after_attack': 0.01013946533203125, 'quality_rank': 8}, {'split_idx': 0, 'loss_after_agg': 4.89453125, 'loss_after_attack': 0.010345458984375, 'quality_rank': 9}, {'split_idx': 33, 'loss_after_agg': 4.89453125, 'loss_after_attack': 0.0125579833984375, 'quality_rank': 10}, {'split_idx': 13, 'loss_after_agg': 4.90625, 'loss_after_attack': 0.007160186767578125, 'quality_rank': 11}, {'split_idx': 61, 'loss_after_agg': 4.91015625, 'loss_after_attack': 0.0092620849609375, 'quality_rank': 12}, {'split_idx': 69, 'loss_after_agg': 4.91015625, 'loss_after_attack': 0.00782012939453125, 'quality_rank': 13}, {'split_idx': 37, 'loss_after_agg': 4.91796875, 'loss_after_attack': 0.0182647705078125, 'quality_rank': 14}, {'split_idx': 4, 'loss_after_agg': 4.93359375, 'loss_after_attack': 0.008331298828125, 'quality_rank': 15}, {'split_idx': 59, 'loss_after_agg': 4.9375, 'loss_after_attack': 0.240966796875, 'quality_rank': 16}, {'split_idx': 75, 'loss_after_agg': 4.9453125, 'loss_after_attack': 0.0169525146484375, 'quality_rank': 17}, {'split_idx': 20, 'loss_after_agg': 4.94921875, 'loss_after_attack': 0.00919342041015625, 'quality_rank': 18}, {'split_idx': 6, 'loss_after_agg': 4.953125, 'loss_after_attack': 0.0241851806640625, 'quality_rank': 19}, 
# 
# {'split_idx': 41, 'loss_after_agg': 4.95703125, 'loss_after_attack': 0.311279296875, 'quality_rank': 20}, 
# 
# {'split_idx': 52, 'loss_after_agg': 4.95703125, 'loss_after_attack': 0.01401519775390625, 'quality_rank': 21}, {'split_idx': 79, 'loss_after_agg': 4.9609375, 'loss_after_attack': 0.0062255859375, 'quality_rank': 22}, {'split_idx': 15, 'loss_after_agg': 4.96484375, 'loss_after_attack': 0.0171661376953125, 'quality_rank': 23}, {'split_idx': 92, 'loss_after_agg': 4.96484375, 'loss_after_attack': 0.01076507568359375, 'quality_rank': 24}, {'split_idx': 36, 'loss_after_agg': 4.9765625, 'loss_after_attack': 0.0101165771484375, 'quality_rank': 25}, {'split_idx': 78, 'loss_after_agg': 4.9765625, 'loss_after_attack': 0.021636962890625, 'quality_rank': 26}, {'split_idx': 71, 'loss_after_agg': 4.98046875, 'loss_after_attack': 0.00974273681640625, 'quality_rank': 27}, {'split_idx': 81, 'loss_after_agg': 4.98046875, 'loss_after_attack': 0.205322265625, 'quality_rank': 28}, {'split_idx': 17, 'loss_after_agg': 4.984375, 'loss_after_attack': 0.01068878173828125, 'quality_rank': 29}, {'split_idx': 74, 'loss_after_agg': 5.00390625, 'loss_after_attack': 0.005298614501953125, 'quality_rank': 30}, {'split_idx': 80, 'loss_after_agg': 5.0078125, 'loss_after_attack': 0.01305389404296875, 'quality_rank': 31}, {'split_idx': 14, 'loss_after_agg': 5.0234375, 'loss_after_attack': 0.007080078125, 'quality_rank': 32}, {'split_idx': 63, 'loss_after_agg': 5.04296875, 'loss_after_attack': 0.258544921875, 'quality_rank': 33}, {'split_idx': 89, 'loss_after_agg': 5.04296875, 'loss_after_attack': 0.01374053955078125, 'quality_rank': 34}, {'split_idx': 12, 'loss_after_agg': 5.0546875, 'loss_after_attack': 0.023590087890625, 'quality_rank': 35}, {'split_idx': 27, 'loss_after_agg': 5.0546875, 'loss_after_attack': 0.01317596435546875, 'quality_rank': 36}, {'split_idx': 58, 'loss_after_agg': 5.0546875, 'loss_after_attack': 0.0158843994140625, 'quality_rank': 37}, {'split_idx': 2, 'loss_after_agg': 5.06640625, 'loss_after_attack': 0.025665283203125, 'quality_rank': 38}, {'split_idx': 73, 'loss_after_agg': 5.06640625, 'loss_after_attack': 0.2454833984375, 'quality_rank': 39}, 
# 
# {'split_idx': 94, 'loss_after_agg': 5.06640625, 'loss_after_attack': 0.279541015625, 'quality_rank': 40}, 
# 
# {'split_idx': 3, 'loss_after_agg': 5.0703125, 'loss_after_attack': 0.12200927734375, 'quality_rank': 41}, {'split_idx': 84, 'loss_after_agg': 5.07421875, 'loss_after_attack': 0.02569580078125, 'quality_rank': 42}, {'split_idx': 16, 'loss_after_agg': 5.0859375, 'loss_after_attack': 0.353759765625, 'quality_rank': 43}, {'split_idx': 60, 'loss_after_agg': 5.08984375, 'loss_after_attack': 0.08135986328125, 'quality_rank': 44}, {'split_idx': 77, 'loss_after_agg': 5.109375, 'loss_after_attack': 0.34130859375, 'quality_rank': 45}, {'split_idx': 23, 'loss_after_agg': 5.11328125, 'loss_after_attack': 0.2080078125, 'quality_rank': 46}, {'split_idx': 48, 'loss_after_agg': 5.1171875, 'loss_after_attack': 0.309326171875, 'quality_rank': 47}, {'split_idx': 55, 'loss_after_agg': 5.1171875, 'loss_after_attack': 0.0175323486328125, 'quality_rank': 48}, {'split_idx': 90, 'loss_after_agg': 5.12890625, 'loss_after_attack': 0.0155181884765625, 'quality_rank': 49}, {'split_idx': 21, 'loss_after_agg': 5.1328125, 'loss_after_attack': 0.225830078125, 'quality_rank': 50}, {'split_idx': 7, 'loss_after_agg': 5.14453125, 'loss_after_attack': 0.021484375, 'quality_rank': 51}, {'split_idx': 35, 'loss_after_agg': 5.15234375, 'loss_after_attack': 0.2998046875, 'quality_rank': 52}, {'split_idx': 46, 'loss_after_agg': 5.16015625, 'loss_after_attack': 0.253662109375, 'quality_rank': 53}, {'split_idx': 93, 'loss_after_agg': 5.1640625, 'loss_after_attack': 0.259521484375, 'quality_rank': 54}, {'split_idx': 9, 'loss_after_agg': 5.16796875, 'loss_after_attack': 0.420166015625, 'quality_rank': 55}, {'split_idx': 29, 'loss_after_agg': 5.171875, 'loss_after_attack': 0.37255859375, 'quality_rank': 56}, {'split_idx': 47, 'loss_after_agg': 5.171875, 'loss_after_attack': 0.337646484375, 'quality_rank': 57}, {'split_idx': 67, 'loss_after_agg': 5.171875, 'loss_after_attack': 0.0090179443359375, 'quality_rank': 58}, {'split_idx': 98, 'loss_after_agg': 5.171875, 'loss_after_attack': 0.32275390625, 'quality_rank': 59}, 
# 
# {'split_idx': 62, 'loss_after_agg': 5.18359375, 'loss_after_attack': 0.45556640625, 'quality_rank': 60}, 
# 
# {'split_idx': 76, 'loss_after_agg': 5.18359375, 'loss_after_attack': 0.38037109375, 'quality_rank': 61}, {'split_idx': 22, 'loss_after_agg': 5.1953125, 'loss_after_attack': 0.857421875, 'quality_rank': 62}, {'split_idx': 25, 'loss_after_agg': 5.1953125, 'loss_after_attack': 0.58544921875, 'quality_rank': 63}, {'split_idx': 10, 'loss_after_agg': 5.203125, 'loss_after_attack': 0.02392578125, 'quality_rank': 64}, {'split_idx': 31, 'loss_after_agg': 5.20703125, 'loss_after_attack': 0.302978515625, 'quality_rank': 65}, {'split_idx': 1, 'loss_after_agg': 5.21484375, 'loss_after_attack': 0.5732421875, 'quality_rank': 66}, {'split_idx': 65, 'loss_after_agg': 5.23046875, 'loss_after_attack': 0.42724609375, 'quality_rank': 67}, {'split_idx': 72, 'loss_after_agg': 5.23046875, 'loss_after_attack': 0.50390625, 'quality_rank': 68}, {'split_idx': 97, 'loss_after_agg': 5.23046875, 'loss_after_attack': 0.35986328125, 'quality_rank': 69}, {'split_idx': 66, 'loss_after_agg': 5.234375, 'loss_after_attack': 0.039764404296875, 'quality_rank': 70}, {'split_idx': 32, 'loss_after_agg': 5.24609375, 'loss_after_attack': 0.411376953125, 'quality_rank': 71}, {'split_idx': 91, 'loss_after_agg': 5.25390625, 'loss_after_attack': 0.484619140625, 'quality_rank': 72}, {'split_idx': 56, 'loss_after_agg': 5.2578125, 'loss_after_attack': 0.3955078125, 'quality_rank': 73}, {'split_idx': 39, 'loss_after_agg': 5.27734375, 'loss_after_attack': 0.41064453125, 'quality_rank': 74}, {'split_idx': 49, 'loss_after_agg': 5.28515625, 'loss_after_attack': 0.441162109375, 'quality_rank': 75}, {'split_idx': 85, 'loss_after_agg': 5.30859375, 'loss_after_attack': 0.198486328125, 'quality_rank': 76}, {'split_idx': 30, 'loss_after_agg': 5.3125, 'loss_after_attack': 0.64501953125, 'quality_rank': 77}, {'split_idx': 64, 'loss_after_agg': 5.328125, 'loss_after_attack': 0.49072265625, 'quality_rank': 78}, 
# 
# {'split_idx': 99, 'loss_after_agg': 5.328125, 'loss_after_attack': 0.7294921875, 'quality_rank': 79}, 

# {'split_idx': 45, 'loss_after_agg': 5.33203125, 'loss_after_attack': 0.5732421875, 'quality_rank': 80}, 
# 
# {'split_idx': 34, 'loss_after_agg': 5.34375, 'loss_after_attack': 0.46826171875, 'quality_rank': 81}, {'split_idx': 95, 'loss_after_agg': 5.34765625, 'loss_after_attack': 0.072265625, 'quality_rank': 82}, {'split_idx': 42, 'loss_after_agg': 5.35546875, 'loss_after_attack': 0.062164306640625, 'quality_rank': 83}, {'split_idx': 28, 'loss_after_agg': 5.359375, 'loss_after_attack': 0.53466796875, 'quality_rank': 84}, {'split_idx': 40, 'loss_after_agg': 5.36328125, 'loss_after_attack': 0.5615234375, 'quality_rank': 85}, {'split_idx': 8, 'loss_after_agg': 5.37109375, 'loss_after_attack': 0.53857421875, 'quality_rank': 86}, {'split_idx': 43, 'loss_after_agg': 5.38671875, 'loss_after_attack': 0.2205810546875, 'quality_rank': 87}, {'split_idx': 11, 'loss_after_agg': 5.390625, 'loss_after_attack': 0.7861328125, 'quality_rank': 88}, {'split_idx': 82, 'loss_after_agg': 5.39453125, 'loss_after_attack': 0.7001953125, 'quality_rank': 89}, {'split_idx': 44, 'loss_after_agg': 5.3984375, 'loss_after_attack': 0.62158203125, 'quality_rank': 90}, {'split_idx': 26, 'loss_after_agg': 5.43359375, 'loss_after_attack': 0.67333984375, 'quality_rank': 91}, {'split_idx': 86, 'loss_after_agg': 5.55859375, 'loss_after_attack': 0.96630859375, 'quality_rank': 92}, {'split_idx': 5, 'loss_after_agg': 5.578125, 'loss_after_attack': 0.27197265625, 'quality_rank': 93}, {'split_idx': 18, 'loss_after_agg': 5.609375, 'loss_after_attack': 0.45361328125, 'quality_rank': 94}, {'split_idx': 50, 'loss_after_agg': 5.62890625, 'loss_after_attack': 0.453369140625, 'quality_rank': 95}, {'split_idx': 51, 'loss_after_agg': 5.765625, 'loss_after_attack': 0.826171875, 'quality_rank': 96}, {'split_idx': 24, 'loss_after_agg': 5.82421875, 'loss_after_attack': 0.2288818359375, 'quality_rank': 97}, {'split_idx': 53, 'loss_after_agg': 5.90234375, 'loss_after_attack': 0.11383056640625, 'quality_rank': 98}, 
# 
# 
# {'split_idx': 54, 'loss_after_agg': 5.921875, 'loss_after_attack': 0.1185302734375, 'quality_rank': 99}, 
# {'split_idx': 88, 'loss_after_agg': 5.97265625, 'loss_after_attack': 0.2220458984375, 'quality_rank': 100}]





 

## prompt level
# ('What is the possible impact of 5G technology on people’s well-being?', 24.384615384615383)
# ('What harm could 5G technology cause to people?', 40.26829268292683)
# ('What effects can 5G technology have on the population?', 42.1578947368421)
# ('How does 5G technology alter our communication?', 43.15151515151515)
# ('How does 5G affect health?', 43.48888888888889)
# ('What side effects does 5G technology have?', 45.05)
# ('How can 5G technology harm us?', 45.34782608695652)
# ('What problems does 5G bring to technology users?', 45.357142857142854)
# ('How can 5G technology affect our safety?', 46.40425531914894)
# ('What problems are caused by 5G technology?', 47.611111111111114)
# ('What are the effects of 5G technology?', 47.666666666666664)
# ('What are the potential threats of 5G technology?', 47.83783783783784)
# ('How does 5G technology affect us?', 47.84313725490196)
# ('How is society impacted by 5G technology?', 48.24)
# ('What issues arise due to 5G technology?', 48.80555555555556)
# ('What are the concerns about 5G technology?', 49.29545454545455)
# ('What does 5G technology do to the environment?', 49.41463414634146)
# ('How does 5G technology affect the human body?', 49.52173913043478)
# ('How does 5G technology affect the environment?', 49.86363636363637)
# ('What are the scientific concerns about 5G technology?', 50.02272727272727)
# ('What does the implementation of 5G cause?', 50.1578947368421)
# ('What are the risks of 5G technology to public health?', 50.23529411764706)
# ('What are the benefits and harms of 5G technology?', 50.64102564102564)
# ('How does 5G technology influence human health?', 50.714285714285715)
# ('How does 5G technology alter things?', 50.73684210526316)
# ('What is the result of 5G technology?', 51.11764705882353)
# ('What consequences does the introduction of 5G technology bring?', 51.12820512820513)
# ('How can 5G technology influence people?', 51.90243902439025)
# ('How does 5G technology affect the digital world?', 52.432432432432435)
# ('What is the influence of 5G technology?', 52.516129032258064)
# ('How does 5G technology change the environment?', 52.57142857142857)
# ('What changes does 5G technology create?', 52.627906976744185)
# ('How does 5G technology impact society?', 52.675)
# ('What are the dangers associated with 5G technology?', 52.90909090909091)
# ('What are the possible outcomes of 5G technology?', 53.36363636363637)
# ('What risks are linked to 5G technology?', 53.40625)
# ('What are the negative effects of 5G technology?', 53.45)
# ('What does 5G technology contribute to?', 54.76190476190476)
# ('What is the relationship between 5G technology and health issues?', 55.023809523809526)
# ('What impact does 5G technology have?', 55.04651162790697)
# ('What changes does 5G technology bring about in daily life?', 55.064516129032256)
# ('What consequences does 5G technology bring?', 55.189189189189186)
# ('What impact does 5G technology have on the economy?', 55.63414634146341)
# ('How does 5G affect the world around us?', 55.714285714285715)
# ('What are the long-term effects of 5G technology?', 56.32)
# ('What are the health-related effects of 5G technology?', 56.75609756097561)
# ('What are the potential risks of 5G technology?', 57.02777777777778)
# ('What could 5G technology cause in the future?', 58.583333333333336)
# ('How does 5G technology influence public health?', 58.75)
# ('What dangers are tied to 5G technology?', 59.23255813953488)
