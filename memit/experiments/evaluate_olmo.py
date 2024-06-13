import json
import shutil
from itertools import islice
from time import time
import random
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from dataclasses import replace
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    #"FT": (FTHyperParams, apply_ft_to_model),
    #"MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

def evaluate(model, tokenizer, concept, location, QA, text_completion, unrelated_QA, top_k=200):
    #evaluate on Cosine similarity, Jaccard Similarity, QA, text_completion

    qa_answers = []
    text_responses = []
    unrelated_qa_answers = []
    E = model.get_output_embeddings().weight.detach() #should use the new projection by the new model's lm_head
    layer, dim = location
    if 'llama' in model.config.model_type:
        params = model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'].T[dim, :]
    elif 'olmo' in model.config.model_type:
        params = model.state_dict()[f'model.transformer.blocks.{layer}.ff_out.weight'].T[dim, :]


    logits = params.T.matmul(E.T)
    _, sorted_indices_item = torch.sort(logits, descending=True)
    ids = [i.item() for i in sorted_indices_item[:top_k]]
    projection = [tokenizer._convert_id_to_token(i) for i in ids]

    QA = [f"Question: {question} Answer: " for question in QA]
    inputs = tokenizer(QA, return_tensors="pt", padding=True, return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        generation_output = model.generate(  # mt.model
            **inputs,
            do_sample=False,
            max_new_tokens=200,
        )
    qa_answers= tokenizer.batch_decode(generation_output[:, -200:], skip_special_tokens=True)

    text_completion = [f"Please complete the following paragraph: {text['First_half']}" for text in text_completion]
    inputs = tokenizer(text_completion, return_tensors="pt", padding=True, return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        generation_output = model.generate(  # mt.model
            **inputs,
            do_sample=False,
            max_new_tokens=200,
        )
    text_responses= tokenizer.batch_decode(generation_output[:, -200:], skip_special_tokens=True)

    unrelated_QA = [f"Question: {unrelated_question} Answer: " for unrelated_question in unrelated_QA] ##Testing its normal ability
    inputs = tokenizer(unrelated_QA, return_tensors="pt", padding=True, return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        generation_output = model.generate(  # mt.model
            **inputs,
            do_sample=False,
            max_new_tokens=200,
        )
    unrelated_qa_answers = tokenizer.batch_decode(generation_output[:, -200:], skip_special_tokens=True)


    return params, projection, qa_answers, text_responses, unrelated_qa_answers

def main(
    args,
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    order: int = 0,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    #print("Loading dataset, attribute snippets, tf-idf data")
    # snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    # vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None



    # ds_class, ds_eval_method = DS_DICT[ds_name]
    # ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    results_save_path = '/U_PZL2023ZZ0005/yhhong/unlearn_results/olmo-7b/memit/entropy/'
    with open("/home/yihuaihong/Unlearn_Harry_Potter/ConceptMap/ConceptMap_data/relation_for_KE/relation_to_template.json", "r",
              encoding="utf-8") as file1:
        relation_to_template = json.load(file1)

    with open("/home/yihuaihong/Unlearn_Harry_Potter/ConceptMap/ConceptMap_data/olmo-7b_concepts_test.json", "r",
              encoding="utf-8") as file2:
        concepts_list = json.load(file2)

    with open("/home/yihuaihong/Unlearn_Harry_Potter/ConceptMap/ConceptMap_data/relation_for_KE/olmo_relation_object_test.json", "r",
              encoding="utf-8") as file3:
        concepts = json.load(file3)

    results = []
    order = args.order
    subject = concepts[order]['subject']
    print(f'editing on {order}_{subject}..')

    assert concepts_list[order]['Concept'] == subject
    location = (concepts_list[order]['Layer'], concepts_list[order]['Dim'])
    hparams = replace(hparams, layers=[layer for layer in range(location[0]-3, location[0]+1)])

    relations = concepts[order]['relation']
    num_edits = len(relations)
    args.num_edits = num_edits
    print(f'MEMIT unlearning data preparing... editing {num_edits} relations in the same time..')
    ds = []
    for ix, relation in enumerate(relations):

        #limit relations number
        if ix >=50:
            break

        ds.append(
            {
                "case_id": ix,
                "requested_rewrite": {
                    "prompt": relation_to_template[relation[0]]['template'],
                    "relation_id": relation[0],
                    "target_new": {
                        "str": relation[1]
                    },
                    "target_true": {
                        "str": relation[1]
                    },
                    "subject": subject
                }}
            # {
            #     "case_id": ix,
            #     "requested_rewrite": {
            #         "prompt": "The capital city of {} is",
            #         "relation_id": relation[0],
            #         "target_new": {
            #             "str": "London"
            #         },
            #         "target_true": {
            #             "str": "London"
            #         },
            #         "subject": "Poland"
            #     }}
        )



    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")


    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-order_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, order)
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        #case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        # print('args.fact_erasure: ',args.fact_erasure)

        args.dummy_string = False
        args.fact_erasure = True
        args.entropy_loss = True

        if args.dummy_string:
            args.fact_erasure = False
            args.margin_loss = False
            args.entropy_loss = False
            print('running on dummy string editing...')



        edited_model, weights_copy = apply_algo(
            args,
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model

        unrelated_QA = [item for sublist in random.sample([random.sample(x['QA'], 4) for x in concepts_list if x['Concept'] != concepts_list[order]['Concept']], 5) for item in sublist]
        params, projection, qa_answers, text_responses, unrelated_qa_answers = evaluate(model=edited_model, tokenizer=tok,
                                                                                        concept=concepts_list[order]['Concept'],
                                                                                        location=location, QA=concepts_list[order]['QA'],
                                                                                        text_completion=concepts_list[order]['text_completion'],
                                                                                        unrelated_QA=unrelated_QA)

        print('qa_answers: ',qa_answers)
        print('unrelated_qa_answers: ',unrelated_qa_answers)

        results.append(
            {'id': order, 'Concept': concepts_list[order]['Concept'], 'params': params, 'projection': projection, 'qa_answers': qa_answers,
             'text_responses': text_responses, 'unrelated_qa_answers': unrelated_qa_answers, 'unrelated_QA': unrelated_QA})
        torch.save(results, results_save_path + f"{args.alg_name}_olmo_relation_object_test_concept{order}.pt")

        """
        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
        """


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        #choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="/U_PZL2023ZZ0005/yhhong/transformers/OLMo-7B",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="olmo-7b.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["conceptvectors"],
        default="conceptvectors",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=0,
        help="order on conceptvectors",
        required=True,
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--dummy_string",
        default=False,
        action="store_true",
        help="dummy_string",
    )
    parser.add_argument(
        "--fact_erasure",
        default=True,
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--margin_loss",
        dest="margin_loss",
        action="store_true",
        help="Add margin loss",
    )
    parser.add_argument(
        "--entropy_loss",
        dest="entropy_loss",
        action="store_true",
        help="Add entropy loss",
    )
    parser.add_argument(
        "--margin_layers",
        dest="margin_layers",
        default=[23,24,25,26,27,28,29,30,31,32],
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--entropy_layers",
        dest="entropy_layers",
        default=[23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=4,
        help="top-k metric",
    )

    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args=args,
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        continue_from_run=args.continue_from_run,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )
