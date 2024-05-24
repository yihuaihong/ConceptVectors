import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn.functional as F
from rouge import Rouge
# from bert_score import score
from nltk.corpus import stopwords
stopwords0_ = stopwords.words('english')

def evaluate(model, tokenizer, QA, text_completion, unrelated_QA):
    # evaluate on Cosine similarity, Jaccard Similarity, QA, text_completion

    qa_answers = []
    text_responses = []
    unrelated_qa_answers = []

    for question in QA:
        inputs = tokenizer(f"Question: {question} Answer: ", return_tensors="pt")
        # print('inputs: ',inputs)
        input_ids = inputs["input_ids"].to('cuda')

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                do_sample=False,
                max_new_tokens=200,
            )
        s = generation_output.sequences[0]
        qa_answers.append(tokenizer.decode(s))

    for text in text_completion:
        inputs = tokenizer(f"Please complete the following paragraph: text['First_half']", return_tensors="pt")
        # print('inputs: ',inputs)
        input_ids = inputs["input_ids"].to('cuda')

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                do_sample=False,
                max_new_tokens=200,
            )
        s = generation_output.sequences[0]
        text_responses.append(tokenizer.decode(s))

    for question in unrelated_QA:  # Testing its normal ability
        inputs = tokenizer(f"Question: {question} Answer: ", return_tensors="pt")
        # print('inputs: ',inputs)
        input_ids = inputs["input_ids"].to('cuda')

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                do_sample=False,
                max_new_tokens=200,
            )
        s = generation_output.sequences[0]
        unrelated_qa_answers.append(tokenizer.decode(s))

    return qa_answers, text_responses, unrelated_qa_answers

def jaccard_similarity(model, tokenizer, params1, params2=None, projection2=None, preciser_jaccard=False, wikipedia_content=None):
    top_k = 200
    E = model.get_output_embeddings().weight.detach()

    logits1 = params1.T.matmul(E.T)
    _, sorted_indices_item1 = torch.sort(logits1, descending=True)
    ids1 = [i.item() for i in sorted_indices_item1[:top_k]]
    projection1 = [tokenizer._convert_id_to_token(i) for i in ids1]

    if params2 is None:
        logits2 = params2.T.matmul(E.T)
        _, sorted_indices_item2 = torch.sort(logits2, descending=True)
        ids2 = [i.item() for i in sorted_indices_item2[:top_k]]
        projection2 = [tokenizer._convert_id_to_token(i) for i in ids2]

    set1 = set(projection1)
    set2 = set(projection2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union

    if preciser_jaccard is True:

        tokens = tokenizer.tokenize(wikipedia_content)
        tokens = [token for token in tokens if token.lower() not in stopwords0_]  #tokens appear in wikipedia

        token_freq = {} #tokens appear in wikipedia
        for token in tokens:
            if token in token_freq:
                token_freq[token] += 1
            else:
                token_freq[token] = 1

        filter_projection1 = [token for token in projection1 if token in token_freq]
        left_projection2 = [token for token in projection2 if token in filter_projection1]

        preciser_sim = len(left_projection2) / len(projection2)


    return similarity, preciser_sim


def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    cc = SmoothingFunction()
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=cc.method3)
    return bleu_score


# Rouge-L Score Calculation
def calculate_rouge_l(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score


def cosine_similarity(params1, params2):
    similarity = F.cosine_similarity(params1, params2, dim=-1).item()

    # make sure that the cosine_similarity<=1
    similarity = min(similarity, 1.0)

    return similarity


def norm_distance(params1, params2):
    euclidean_distance = torch.norm(params1 - params2)
    return euclidean_distance.item()