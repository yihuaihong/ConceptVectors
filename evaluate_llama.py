import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
# from bert_score import score
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
import re

from tqdm import tqdm
#from transformers_source.src.transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

torch.set_grad_enabled(False)
tqdm.pandas()

# Utilities
from utils import set_random_seed
from evaluate_util import evaluate, cosine_similarity, jaccard_similarity, norm_distance, calculate_rouge_l, calculate_bleu


# device = 'cuda:1'
set_random_seed(8888)
loss_type = "dpo"
ft_type = "Needle"
running_set = "dev"


def merge_pt_files(base_dir, num_files):
    final_list = []

    for i in range(num_files):
        #file_name = f'llama2-7b_concepts_results_grad_ascent_NiddleFull_test_concept{i}.pt'
        file_name = f'llama2-7b_concepts_results_{loss_type}_{ft_type}_{running_set}_concept{i}.pt'

        file_path = os.path.join(base_dir, file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in directory {base_dir}")

        # Load the list from the pt file
        loaded_list = torch.load(file_path)
        # Extend the final_list with elements from the loaded list
        final_list.extend(loaded_list)

    assert len(final_list) == num_files

    return final_list


base_model = '/root/autodl-tmp/transformers/llama2-7b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy = True)
tokenizer.pad_token = tokenizer.eos_token

original_model = LlamaForCausalLM.from_pretrained(base_model).cuda()

base_dir = f'/root/autodl-tmp/unlearn_results/llama2-7b/{loss_type}'
num_files = 95 #95 for llama, 162 for olmo  #will uppdate here
loaded_data = merge_pt_files(base_dir, num_files)

data_path = '/root/Unlearn_Harry_Potter/Baselines/ConceptVectors/ConceptVectors_data'
with open(data_path + f"/llama2-7b_concepts_{running_set}.json", "r", encoding="utf-8") as file:
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
sum_related_tokens_rate = []

output_file = base_dir+f"/evaluation_{running_set}_results.txt"
with open(output_file, "w") as f:
    f.write("intermediate results: \n")

for ix, (item, concept) in enumerate(zip(loaded_data, concept_data)):
    print(f"evaluating on {ix}_concept about {item['Concept']}")


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

    unrelated_QA = concept['unrelated_QA']


    original_qa_answers, original_text_responses, original_unrelated_qa_answers = evaluate(original_model, tokenizer, QA, text_completion,
                                                                unrelated_QA)


    for idy, (unlearn_qa_answer, qa_answer) in enumerate(zip(unlearn_qa_answers, original_qa_answers)):
        if idy == 0:
            print(f"unlearn_qa_answer: {unlearn_qa_answer}\n qa_answer: {qa_answer}\n")
        qa_bleu_scores.append(calculate_bleu(unlearn_qa_answer, qa_answer))
        qa_rouge_l_scores.append(calculate_rouge_l(unlearn_qa_answer, qa_answer))

    for idy, (unlearn_text_response, text_response) in enumerate(zip(unlearn_text_responses, original_text_responses)):
        if idy == 0:
            print(f"unlearn_text_response: {unlearn_text_response}\n text_response: {text_response}\n")
        text_bleu_scores.append(calculate_bleu(unlearn_text_response, text_response))
        text_rouge_l_scores.append(calculate_rouge_l(unlearn_text_response, text_response))

    for idy, (unlearn_unrelated_qa_answer, unrelated_qa_answer) in enumerate(zip(unlearn_unrelated_qa_answers,
                                                                original_unrelated_qa_answers)):
        if idy == 0:
            print(f"unlearn_unrelated_qa_answer: {unlearn_unrelated_qa_answer}\n unrelated_qa_answer: {unrelated_qa_answer}\n")
        unrelated_bleu_scores.append(calculate_bleu(unlearn_unrelated_qa_answer, unrelated_qa_answer))
        unrelated_rouge_l_scores.append(calculate_rouge_l(unlearn_unrelated_qa_answer, unrelated_qa_answer))

    qa_bleu_score = statistics.mean(qa_bleu_scores)
    qa_rouge_l_score = statistics.mean(qa_rouge_l_scores)

    text_bleu_score = statistics.mean(text_bleu_scores)
    text_rouge_l_score = statistics.mean(text_rouge_l_scores)

    unrelated_bleu_score = statistics.mean(unrelated_bleu_scores)
    unrelated_rouge_l_score = statistics.mean(unrelated_rouge_l_scores)

    layer, dim = concept['Layer'], concept['Dim']

    sub_tensor1 = original_model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'][:, dim].cuda()
    sub_tensor2 = item['params'].cuda()
    projection2 = item['projection']

    cosine_sim = cosine_similarity(sub_tensor1, sub_tensor2)
    norm_dis = norm_distance(sub_tensor1, sub_tensor2)
    jaccard_sim, preciser_jaccard_sim, related_tokens_rate = jaccard_similarity(original_model, tokenizer, params1=sub_tensor1, params2=sub_tensor2, projection2=projection2, preciser_jaccard = True, wikipedia_content=concept['wikipedia_content'])

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
    print('related_tokens_rate: ', related_tokens_rate)

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
    sum_related_tokens_rate.append(related_tokens_rate)

    # Write intermediate results to file
    with open(output_file, "a") as f:
        f.write(f'Concept {ix} Results\n')
        f.write(f'qa_bleu_score: {qa_bleu_score}\n')
        f.write(f'qa_rouge_l_score: {qa_rouge_l_score}\n')
        f.write(f'text_bleu_score: {text_bleu_score}\n')
        f.write(f'text_rouge_l_score: {text_rouge_l_score}\n')
        f.write(f'unrelated_bleu_score: {unrelated_bleu_score}\n')
        f.write(f'unrelated_rouge_l_score: {unrelated_rouge_l_score}\n')
        f.write(f'cosine_similarity: {cosine_sim}\n')
        f.write(f'norm_distance: {norm_dis}\n')
        f.write(f'jaccard_similarity: {jaccard_sim}\n')
        f.write(f'preciser_jaccard_similarity: {preciser_jaccard_sim}\n')
        f.write(f'related_tokens_rate: {related_tokens_rate}\n')
        f.write('\n')


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
print(f"Avg_related_tokens_rate on {ix+1} concepts is {statistics.mean(sum_related_tokens_rate)}")


with open(output_file, "a") as f:
    f.write("Final Normal Evaluation Results\n")
    f.write(f"Avg_qa_bleu_score on {ix+1} concepts is {statistics.mean(sum_qa_bleu_scores)}\n")
    f.write(f"Avg_qa_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_qa_rouge_l_scores)}n")
    f.write(f"Avg_text_bleu_score on {ix+1} concepts is {statistics.mean(sum_text_bleu_scores)}\n")
    f.write(f"Avg_text_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_text_rouge_l_scores)}\n")
    f.write(f"Avg_unrelated_bleu_score on {ix+1} concepts is {statistics.mean(sum_unrelated_bleu_scores)}\n")
    f.write(f"Avg_unrelated_rouge_l_score on {ix+1} concepts is {statistics.mean(sum_unrelated_rouge_l_scores)}\n")

    f.write("\nFinal Parameter-based Evaluation Results\n")
    f.write(f"Avg_cosine_similarity on {ix+1} concepts is {statistics.mean(sum_cosine_sim)}\n")
    f.write(f"Avg_norm_distance on {ix+1} concepts is {statistics.mean(sum_norm_dis)}\n")
    f.write(f"Avg_jaccard_similarity on {ix+1} concepts is {statistics.mean(sum_jaccard_sim)}\n")
    f.write(f"Avg_preciser_jaccard_similarity on {ix+1} concepts is {statistics.mean(sum_preciser_jaccard_sim)}\n")
    f.write(f"Avg_related_tokens_rate on {ix + 1} concepts is {statistics.mean(sum_related_tokens_rate)}\n")
