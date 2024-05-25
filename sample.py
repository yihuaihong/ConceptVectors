import random
import json
random.seed(999)

data_path = 'E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/llama2-7b-chat_concepts'

with open(data_path +"/llama_concepts.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    print(len(data))

    dev_set = random.sample(data, int(len(data) *0.1))  # 1:9 split dev and set

    # Step 3: Create the second part by excluding elements in the first part
    test_set = [item for item in data if item not in dev_set]

for item in dev_set:
    print(item['Concept'])
