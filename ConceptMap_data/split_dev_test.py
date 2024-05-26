import json
import random
random.seed(999)

with open("/ConceptMap_data/olmo-7b_concepts.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    dev_set = random.sample(data, int(len(data) * 0.1))  # 1:9 split dev and set
    test_set = [item for item in data if item not in dev_set]

# 假设你已经得到了 dev_set 和 test_set

# 保存验证集到文件
with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/olmo_concepts_dev.json", "w", encoding="utf-8") as dev_file:
    json.dump(dev_set, dev_file, ensure_ascii=False, indent=4)

# 保存测试集到文件
with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/olmo_concepts_test.json", "w", encoding="utf-8") as test_file:
    json.dump(test_set, test_file, ensure_ascii=False, indent=4)
