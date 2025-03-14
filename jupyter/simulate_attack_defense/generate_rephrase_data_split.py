

prompt = "What does 5G technology cause?" 

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



nb_split = 100
nb_sub_data = 20


import random
import json
import os

target_dir = "/opt/data/zx/knowledge_manipulation_attack/data/rephrase_split"

for i in range(nb_split):
    split_data = random.sample(rephrase, nb_sub_data)
    
    save_data = [
        {
            "prompt": prompt,
            "rephrase": split_data
        }
    ]
        
    
    
    # print(split_data[:2])
    f = open(os.path.join(target_dir, f"split_{i}.json"), "w")
    json.dump(save_data, f)

# python simulate_attack_defense/generate_rephrase_data_split.py


