import json
import random

with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/llama2-7b_concepts.json", "r", encoding="utf-8") as file1:
    concepts_original = json.load(file1)

with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/relation_for_KE/relation_to_template.json", "r", encoding="utf-8") as file2:
    relation_to_template = json.load(file2)

with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/relation_for_KE/olmo_relation_object_test.json", "r", encoding="utf-8") as file3:
    concepts = json.load(file3)

    random_sample_concept = random.sample(concepts_original,20)

    for sample in random_sample_concept:
        print(sample['Concept'])
        print("QA: ",sample['QA'])

    # num_relations = 0
    # for ix, item in enumerate(concepts):
    #     num_relations += len(item['relation'])
    #     # if len(relations) == 2:
    #     #     print(item)
    # print(f'avg_relations: {num_relations/ix+1}')
#     assert len(concepts) == len(concepts_original)
#     for ix, item in enumerate(concepts):
#         print(item)
#         concept_name = item['subject']
#
#         #print(f"concept_name: {concept_name}, original_concept_name: {concepts_original[ix]['Concept']}")
#         if concept_name != concepts_original[ix]["Concept"]:
#             concepts[ix]['subject'] = concepts_original[ix]["Concept"]
#
#         new_relations = []
#         relations = item['relation']
#         for iy, relation in enumerate(relations):
#             if relation[0] not in relation_to_template:
#                 continue
#             elif relation[0] in ['P1449','P1448','P1559']:
#                 relation[1] = relation[1]['text']
#                 new_relations.append(relation)
#             elif isinstance(relation[1], dict):
#                 continue
#             else:
#                 new_relations.append(relation)
#
#         concepts[ix]['relation'] = new_relations
#
#
# with open("new_olmo_relation_object.json", "w", encoding="utf-8") as file:
#     json.dump(concepts, file, ensure_ascii=False, indent=4)