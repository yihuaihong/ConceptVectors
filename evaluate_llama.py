import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score
import statistics
from ast import literal_eval
import functools
import json
import os
import random
import wget
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
#from transformers_source.src.transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

torch.set_grad_enabled(False)
tqdm.pandas()

# Utilities
from utils import set_random_seed
from evaluate_util import evaluate, cosine_similarity, jaccard_similarity, norm_distance, calculate_rouge_l, calculate_bleu


torch.cuda.set_device(0)
set_random_seed(1001)

base_model = '/root/autodl-tmp/transformers/llama2-7b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy = True)
original_model = LlamaForCausalLM.from_pretrained(base_model).to('cuda')

loaded_data = torch.load('/root/autodl-tmp/unlearn_results/merged_llama_concepts_results.pt')
data_path = '/root/Unlearn_Harry_Potter/Baselines/ConceptMap/ConceptMap_data/llama2-7b-chat_concepts'
with open(data_path + "/llama_concepts.json", "r", encoding="utf-8") as file:
    concept_data = json.load(file)


sum_qa_bleu_scores = []
sum_qa_rouge_l_scores = []
sum_text_bleu_scores = []
sum_text_rouge_l_scores = []
sum_unrelated_bleu_scores = []
sum_unrelated_rouge_l_scores = []
sum_cosine_sim = []
sum_norm_dis = []
sum_jaccard_sim = []
sum_preciser_jaccard_sim = []

for ix, (item, concept) in enumerate(zip(loaded_data, concept_data)):
    print(f"evaluating on {ix}_concept about {item['Concept']}")

    if ix==25:
       break

    assert item['Concept'] == concept['Concept']

    unlearn_qa_answers = loaded_data[ix]['qa_answers']
    unlearn_text_responses = loaded_data[ix]['text_responses']
    unlearn_unrelated_qa_answers = loaded_data[ix]['unrelated_qa_answers']

    qa_bleu_scores = []
    qa_rouge_l_scores = []
    qa_bert_f_scores = []
    text_bleu_scores = []
    text_rouge_l_scores = []
    text_bert_f_scores = []
    unrelated_bleu_scores = []
    unrelated_rouge_l_scores = []
    unrelated_bert_f_scores = []

    QA = concept['QA']
    text_completion = concept['text_completion']
    unrelated_QA = [item for sublist in random.sample([random.sample(x['QA'], 4) for x in concept_data if x['Concept'] != concept['Concept']], 5) for
                    item in sublist]  # 4questions * 5concepts这个部分需要后面人工调整，unrelated_qa,尽量内容不要有重叠

    original_qa_answers, original_text_responses, original_unrelated_qa_answers = evaluate(original_model, tokenizer, QA, text_completion,
                                                                unrelated_QA)


    for unlearn_qa_answer, qa_answer in zip(unlearn_qa_answers, original_qa_answers):
        qa_bleu_scores.append(calculate_bleu(unlearn_qa_answer, qa_answer))
        qa_rouge_l_scores.append(calculate_rouge_l(unlearn_qa_answer, qa_answer))

    for unlearn_text_response, text_response in zip(unlearn_text_responses, original_text_responses):
        text_bleu_scores.append(calculate_bleu(unlearn_text_response, text_response))
        text_rouge_l_scores.append(calculate_rouge_l(unlearn_text_response, text_response))

    for unlearn_unrelated_qa_answer, unrelated_qa_answer in zip(unlearn_unrelated_qa_answers,
                                                                original_unrelated_qa_answers):
        unrelated_bleu_scores.append(calculate_bleu(unlearn_unrelated_qa_answer, unrelated_qa_answer))
        unrelated_rouge_l_scores.append(calculate_rouge_l(unlearn_unrelated_qa_answer, unrelated_qa_answer))

    qa_bleu_score = statistics.mean(qa_bleu_scores)
    qa_rouge_l_score = statistics.mean(qa_rouge_l_scores)

    # bert_f_scores = [tensor.item() for tensor in bert_f_scores]
    # bert_f_score = statistics.mean(bert_f_scores)
    text_bleu_score = statistics.mean(text_bleu_scores)
    text_rouge_l_score = statistics.mean(text_rouge_l_scores)

    unrelated_bleu_score = statistics.mean(unrelated_bleu_scores)
    unrelated_rouge_l_score = statistics.mean(unrelated_rouge_l_scores)

    layer, dim = concept['Layer'], concept['Dim']

    sub_tensor1 = original_model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'][:, dim].cuda()
    sub_tensor2 = item['params']
    projection2 = item['projection']

    cosine_sim = cosine_similarity(sub_tensor1, sub_tensor2)
    norm_dis = norm_distance(sub_tensor1, sub_tensor2)
    jaccard_sim, preciser_jaccard_sim = jaccard_similarity(original_model, tokenizer, params1=sub_tensor1, params2=sub_tensor2, projection2=projection2, preciser_jaccard = True, wikipedia_content=concept['wikipedia_content'])

    print('qa_bleu_score: ',qa_bleu_score)
    print('qa_rouge_l_score: ', qa_rouge_l_score)
    print('text_bleu_score: ', text_bleu_score)
    print('text_rouge_l_score: ', text_rouge_l_score)
    print('unrelated_bleu_score: ', unrelated_bleu_score)
    print('unrelated_rouge_l_score: ', unrelated_rouge_l_score)

    print('cosine_similarity: ', cosine_sim)
    print('norm_distance: ', norm_dis)
    print('jaccard_similarity: ', jaccard_sim)
    print('preciser_jaccard_similarity: ', preciser_jaccard_sim)

    sum_qa_bleu_scores.append(qa_bleu_score)
    sum_qa_rouge_l_scores.append(qa_rouge_l_score)
    sum_text_bleu_scores.append(text_bleu_score)
    sum_text_rouge_l_scores.append(text_rouge_l_score)
    sum_unrelated_bleu_scores.append(unrelated_bleu_score)
    sum_unrelated_rouge_l_scores.append(unrelated_rouge_l_score)
    sum_cosine_sim.append(cosine_sim)
    sum_norm_dis.append(norm_dis)
    sum_jaccard_sim.append(jaccard_sim)
    sum_preciser_jaccard_sim.append(preciser_jaccard_sim)



print("Normal Evaluation Results")
print(f"Avg_qa_bleu_score on {ix+1} concepts is {statistics.mean(sum_qa_bleu_scores)}")
print(f"Avg_qa_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_qa_rouge_l_scores)}")
print(f"Avg_text_bleu_score on {ix+1} concepts is {statistics.mean(sum_text_bleu_scores)}")
print(f"Avg_text_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_text_rouge_l_scores)}")
print(f"Avg_unrelated_bleu_score on {ix+1} concepts is {statistics.mean(sum_unrelated_bleu_scores)}")
print(f"Avg_unrelated_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_unrelated_rouge_l_scores)}")

print("Parameter-based Evaluation Results")
print(f"Avg_cosine_similarity on {ix+1} concepts is {statistics.mean(sum_cosine_sim)}")
print(f"Avg_norm_distance on {ix+1} concepts is {statistics.mean(sum_norm_dis)}")
print(f"Avg_jaccard_similarity on {ix+1} concepts is {statistics.mean(sum_jaccard_sim)}")
print(f"Avg_preciser_jaccard_similarity on {ix+1} concepts is {statistics.mean(sum_preciser_jaccard_sim)}")
