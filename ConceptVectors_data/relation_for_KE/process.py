import json
import random

with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/relation_for_KE/relation_to_template.json", "r", encoding="utf-8") as file:
    relation_to_template = json.load(file)

# new_relation_to_template = {}
# for key, value in relation_to_template.items():
#     print(key)
#     new_relation_to_template[value[1]] = {'frequency': value[0],'relation_name': key,'template': value[2]}
#     print(new_relation_to_template[value[1]])
#
# # new_file_name = data_path + "/olmo-7b_concepts_original_answers"
# with open('E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/relation_for_KE/new_relation_to_template.json', "w", encoding="utf-8") as file:
#     json.dump(new_relation_to_template, file, ensure_ascii=False, indent=4)

relation2phrase = {
    'head of government': 'The head of government of {} is',
    'brother': 'The brother of {} is',
    'sister': 'The sister of {} is',
    'sibling': "{}'s siblings are",
    'country': 'The country which {} is associated with is',
    'place of birth': 'The city in which {} was born is',
    'place of death': 'The city in which {} died is',
    'sex or gender': "{}'s gender is",
    'father': 'The father of {} is',
    'mother': 'The mother of {} is',
    'spouse': "{}'s spouse is",
    'country of citizenship': 'The country of citizenship of {} is',
    'continent': 'The continent which {} is part of is',
    'head of state': 'The head of state of {} is',
    'capital': 'The capital of {} is',
    'currency': 'The currency in {} is',
    'position held': 'The position that has been held by {} is',
    'official language': 'The official language of {} is',
    'child': 'The child of {} is',
    'stepfather': 'The stepfather of {} is',
    'stepmother': 'The stepmother of {} is',
    'author': 'The author of {} is',
    'member of sports team': '{} has been a member of a sports team. This team is',
    'director': 'The director of {} is',
    'screenwriter': 'The screenwriter of {} is',
    'alma mater': '{} has been educated at',
    'architect': 'The architect of {} is',
    'composer': 'The composer of {} is',
    'anthem': 'The anthem of {} is',
    'sexual orientation': "{}'s sexual orientation is",
    'editor': 'The editor of {} is',
    'occupation': 'The occupation of {} is',
    'employer': "{}'s employer is",
    'founder': 'The founder of {} is',
    'league': 'The league in which {} plays is',
    'place of burial': 'The country in which {} is buried is',
    'field of work': '{} has been working in the field of',
    'native language': 'The mother tongue of {} is',
    'cast member': "{}'s cast members are",
    'award received': '{} has won the award of',
    'follows': '{} follows',
    'ethnic group': 'The ethnic group which {} is associated with is',
    'religion': 'The religion which {} is associated with is',
    'eye color': 'The eye color of {} is',
    'capital of': '{} is the capital of',
    'number of children': 'The number of children {} has is',
    'uncle': 'The uncle of {} is',
    'aunt': 'The aunt of {} is',
    'date of birth': 'The date in which {} was born in is'
}

our_relations = {
    'head of government': 'P6',
    'brother': 'P7',
    'sister': 'P9',
    'sibling': 'P3373',
    'country': 'P17',
    'place of birth': 'P19',
    'place of death': 'P20',
    'sex or gender': 'P21',
    'father': 'P22',
    'mother': 'P25',
    'spouse': 'P26',
    'country of citizenship': 'P27',
    'continent': 'P30',
    'head of state': 'P35',
    'capital': 'P36',
    'currency': 'P38',
    'position held': 'P39',
    'official language': 'P37',
    'child': 'P40',
    'stepfather': 'P43',
    'stepmother': 'P44',
    'author': 'P50',
    'member of sports team': 'P54',
    'director': 'P57',
    'screenwriter': 'P58',
    'alma mater': 'P69',
    'architect': 'P84',
    'composer': 'P86',
    'anthem': 'P85',
    'sexual orientation': 'P91',
    'editor': 'P98',
    'occupation': 'P106',
    'employer': 'P108',
    'founder': 'P112',
    'league': 'P118',
    'place of burial': 'P119',
    'field of work': 'P101',
    'native language': 'P103',
    'cast member': 'P161',
    'award received': 'P166',
    'follows': 'P155',
    'ethnic group': 'P172',
    'religion': 'P140',
    'eye color': 'P1340',
    'capital of': 'P1376',
    'number of children': 'P1971',
    'uncle': '',
    'aunt': '',
    'date of birth': 'P569',
}


def find_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None


for key, value in our_relations.items():
    if value not in relation_to_template:
        relation_to_template[value] = {
            "relation_name": key,
            "template": relation2phrase[key]}


    if value in relation_to_template:
        relation_to_template[value]["template"] =  relation2phrase[key]

with open("E:/NLP/LLM/LM Unlearning/ConceptMap/ConceptMap_data/relation_for_KE/new_relation_to_template.json", "w", encoding="utf-8") as file:
    json.dump(relation_to_template, file, ensure_ascii=False, indent=4)