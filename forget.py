from data_module import TextForgetDatasetWikipedia, TextForgetDatasetDPOQA, TextForgetDatasetKTOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import hydra 
import transformers
import os
import json
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml, set_random_seed
from sparse_ft_utils import find_topK_grads, clip_grads
from evaluate_util import jailbreak_evaluate
import random
import gc

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, loss_threshold):
        self.loss_threshold = loss_threshold
        # self.best_loss = float('inf')
        # self.wait = 0

    def on_log(self, args, state, control, **kwargs):
        # 获取当前评估损失 for grad_diff
        # current_loss = state.metrics["eval_loss"]
        if not state.log_history or "loss" not in state.log_history[-1]:
            return control

        current_loss = state.log_history[-1]["loss"]

        # 如果loss低于阈值,则停止训练
        if current_loss <= self.loss_threshold:
            control.should_training_stop = True

        return control
        # if not state.log_history:
        #     return control
        #
        # print('state.log_history[-1]: ',state.log_history[-1])
        # return control

        # if current_loss < self.loss_threshold:
        #     control.should_training_stop = True
        # # 如果当前损失小于之前记录的最佳损失
        # if current_loss < self.best_loss:
        #     self.best_loss = current_loss
        #     self.wait = 0
        # else:
        #     self.wait += 1
        #     # 如果等待次数超过阈值，则停止训练
        #     if self.wait >= self.early_stopping_threshold:
        #         control.should_training_stop = True



def reset_model_parameters(model, oracle_model):
    model.load_state_dict(oracle_model.state_dict())



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print('trainable param name: ',name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def evaluate(model, tokenizer, concept, location, QA, text_completion, unrelated_QA, top_k=200):
    #evaluate on Cosine similarity, Jaccard Similarity, QA, text_completion


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

    for ix, question in enumerate(QA):
        QA[ix] = f"Question: {question}\n Answer:"

    for ix, text in enumerate(text_completion):
        text_completion[ix] = f"Please complete the following paragraph: {text['First_half']}"

    for ix, question in enumerate(unrelated_QA):  # Testing its normal ability
        unrelated_QA[ix] = f"Question: {question}\n Answer:"

    inputs = tokenizer(QA + text_completion + unrelated_QA, return_tensors="pt", padding=True,return_token_type_ids=False).to('cuda')
    n_new_tokens = 100
    with torch.no_grad():
        generation_output = model.generate(  # mt.model
            **inputs,
            do_sample=False,
            max_new_tokens=100,
        )
    outputs = tokenizer.batch_decode(generation_output[:, -n_new_tokens:], skip_special_tokens=True)
    qa_answers = outputs[:len(QA)]
    text_responses = outputs[len(QA):-len(unrelated_QA)]
    unrelated_qa_answers = outputs[-len(unrelated_QA):]
    assert len(qa_answers) == 10 and len(unrelated_qa_answers) == 50

    print('qa_answers[0]: ', qa_answers[0])
    print('unrelated_qa_answers[0]: ',unrelated_qa_answers[0])
    return params, projection, qa_answers, text_responses, unrelated_qa_answers





@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):


    seed = cfg.seed
    set_random_seed(seed)

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    # model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    # model_id = model_cfg["hf_key"]


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    print("forget_loss_type: ",cfg.forget_loss)
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    if os.path.exists(cfg.save_dir):
        print("Directory already exists")
        if not cfg.overwrite_dir:
            exit()

    max_length = 500

    if cfg.forget_loss in ['grad_ascent', 'grad_diff']:
        oracle_model = None
    else:
        oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16,
                                                            trust_remote_code=True).cuda()



    # first get the base model architectur2e
    # if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break

        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    if path_found:
        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

    # now we have a HuggingFace model
    if bool(cfg.gradient_checkpointing) == True:
        print("gradient_checkpointing is True")
        model.gradient_checkpointing_enable()

    with open(cfg.data_path + f"/{cfg.model_family}_concepts_{cfg.set}.json", "r", encoding="utf-8") as file:
        running_set = json.load(file)

    print(f'running on {cfg.set}')

    order = cfg.order
    item = running_set[order]

    results = []
    concept = item['Concept']
    QA = item['QA']
    Text_completion = item['text_completion']
    location = (item['Layer'], item['Dim'])
    wikipedia_content = item['wikipedia_content']

    random.seed(seed+cfg.order)  #要测试这里是不是真的work，特别对于unrelated_QA的影响，是不是真的能抽到不一样的qa
    random_wikipedia_content = random.sample([x['wikipedia_content'] for x in running_set if x['Concept'] != item['Concept']], 1)  #这个部分需要后面人工调整，尽量内容不要有重叠
    #random_wikipedia_content = [data[22]['wikipedia_content']] #super mario
    #unrelated_QA = [item for sublist in random.sample([random.sample(x['QA'], 4) for x in running_set if x['Concept'] != item['Concept']], 5) for item in sublist] #4questions * 5concepts这个部分需要后面人工调整，unrelated_qa,尽量内容不要有重叠
    unrelated_QA = item['unrelated_QA']
    print('len(unrelated_QA): ',len(unrelated_QA))
    print(f'Training on {order}  {concept}:')

    torch_format_dataset = TextForgetDatasetWikipedia(cfg.data_path,
                                                      content=wikipedia_content,
                                                      random_content=random_wikipedia_content,
                                                      tokenizer=tokenizer,
                                                      model_family=cfg.model_family,
                                                      max_length=max_length,
                                                      split=cfg.split,
                                                      loss_type=cfg.forget_loss)


    batch_size = int(cfg.batch_size)
    gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(
        f"The length of dataset: {len(torch_format_dataset)},\nmax_steps: {max_steps},\nbatch_size: {batch_size},\naccumulation_step: {gradient_accumulation_steps}.")

    if isinstance(cfg.eval_steps, int):
        eval_steps = cfg.eval_steps
    elif cfg.eval_steps == 'steps_per_epoch':
        eval_steps = steps_per_epoch
    else:
        raise NotImplementedError("The eval_steps must be an integer or step_per_epoch.")

    if isinstance(cfg.warmup_steps, int):
        warmup_steps = cfg.warmup_steps
    elif cfg.warmup_steps == 'steps_per_epoch':
        warmup_steps = steps_per_epoch
    else:
        raise NotImplementedError("The warmup_steps must be an integer or step_per_epoch.")



    cfg.lr = float(cfg.lr)
    print(f'cfg.lr: ',cfg.lr)
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=float(cfg.lr),
        bf16=True,
        bf16_full_eval=True,
        logging_steps=steps_per_epoch+1,  # do not save the model
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_steps=max_steps + 1000000,  # do not save the model
        ddp_find_unused_parameters=False,
        # deepspeed='config/ds_config.json',
        weight_decay=cfg.weight_decay,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
    )

    if cfg.ft_type == 'Needle':
        #only ft on specific vector's dimension
        layer, dim = location
        if 'llama' in model.config.model_type:
            for name, param in model.named_parameters():
                if "model.embed_tokens.weight" in name:
                    print('name: ', name)
                    param.requires_grad = True
                    gradient_mask = torch.zeros_like(param).cuda()
                    param.register_hook(lambda grad: grad.mul_(gradient_mask))
                elif f"layers.{layer}.mlp.down_proj" in name:
                    print('name: ', name)
                    param.requires_grad = True
                    gradient_mask_mlp = torch.zeros_like(param).cuda()
                    gradient_mask_mlp[:, dim] = 1
                    param.register_hook(lambda grad: grad.mul_(gradient_mask_mlp))
                else:
                    param.requires_grad = False
        elif 'olmo' in model.config.model_type:
            for name, param in model.named_parameters():
                if "model.embed_tokens.weight" in name:
                    print('name: ', name)
                    param.requires_grad = True
                    gradient_mask = torch.zeros_like(param).cuda()
                    param.register_hook(lambda grad: grad.mul_(gradient_mask))
                elif f'blocks.{layer}.ff_out.weight' in name:
                    print('name: ', name)
                    param.requires_grad = True
                    gradient_mask_mlp = torch.zeros_like(param).cuda()
                    gradient_mask_mlp[:, dim] = 1
                    param.register_hook(lambda grad: grad.mul_(gradient_mask_mlp))
                else:
                    param.requires_grad = False


        print(f"Only train on the param in layer{layer}, dim{dim}.")

        def add_noise(model, location, noise_scale=0):
            # Create Gaussian noise
            mean = 0
            std = noise_scale
            shape = (4096,) #both llama7b and olmo7b is 4096
            noise = torch.normal(mean, std, size=shape).cuda()
            layer, dim = location
            if 'llama' in model.config.model_type:
                model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'][:,dim] += noise
            elif 'olmo' in model.config.model_type:
                model.state_dict()[f'model.transformer.blocks.{layer}.ff_out.weight'][:, dim] += noise

            print(f"adding Guassian Noise on the layer{layer}, dim{dim}'s param")

        add_noise(model, location, noise_scale=0.1)

    elif cfg.ft_type == "all_value_vectors":
        print("Only train on all the value vectors.")
        if 'llama' in model.config.model_type:
            for name, param in model.named_parameters():
                if "model.embed_tokens.weight" in name: #bug here
                    param.requires_grad = True
                    gradient_mask = torch.zeros_like(param).cuda()
                    param.register_hook(lambda grad: grad.mul_(gradient_mask))
                elif f"mlp.down_proj" in name:
                    print('name: ', name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

                # if "model.embed_tokens.weight" in name:
                #     print('name: ', name)
                #     param.requires_grad = True
                #     gradient_mask = torch.zeros_like(param).cuda()
                #     param.register_hook(lambda grad: grad.mul_(gradient_mask))
                # elif f"layers.{layer}.mlp.down_proj" in name:
                #     print('name: ', name)
                #     param.requires_grad = True
                #     gradient_mask_mlp = torch.zeros_like(param).cuda()
                #     gradient_mask_mlp[:, dim] = 1
                #     param.register_hook(lambda grad: grad.mul_(gradient_mask_mlp))
                # else:
                #     param.requires_grad = False
        elif 'olmo' in model.config.model_type:
            for name, param in model.named_parameters():
                if f'ff_out.weight' in name:
                    print('name: ', name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    # elif cfg.ft_type == "Sparse":
    #         print('running on the dataset to find the parameters needed to be ft')
    #         torch_format_dataset
    #         min_grad = find_topK_grads(model, topK = 0.001, c_types=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    #
    #                    clip_grads


    # 创建EarlyStoppingCallback对象并传入早停阈值
    if cfg.forget_loss == 'npo' or cfg.forget_loss == 'dpo':
        cfg.loss_threshold = 0
        print('forget_loss is NPO or DPO, so the early stopping loss_threshold = 0')
    early_stopping_callback = EarlyStoppingCallback(loss_threshold=cfg.loss_threshold)


    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        seed=seed,
        ref_policy=cfg.ref_policy,
        beta=cfg.beta,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
        KL_coeff=cfg.KL_coeff,
        callbacks=[early_stopping_callback]
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()



    """
    params, projection, qa_answers, text_responses, unrelated_qa_answers = evaluate(model, tokenizer=tokenizer, concept=concept, location=location, QA=QA, text_completion=Text_completion, unrelated_QA=unrelated_QA)
    results.append({'id': order,'Concept': concept, 'params': params, 'projection': projection, 'qa_answers': qa_answers, 'text_responses': text_responses, 'unrelated_qa_answers': unrelated_qa_answers})

    torch.save(results, cfg.results_save_path+ f"/{cfg.model_family}_concepts_results_{cfg.forget_loss}_{cfg.ft_type}_{cfg.set}_concept{order}.pt")
    """

    #Jailbreak Evalutation
    jailbreak_evaluate(model=model, tokenizer=tokenizer, Concept=item, data=running_set, cfg=cfg)


    # reset_model_parameters(model, oracle_model)
    # del results

    # save the tokenizer
    # model.save_pretrained(cfg.save_dir)
    # tokenizer.save_pretrained(cfg.save_dir)

    # del trainer
    # del training_args
    # del torch_format_dataset
    # gc.collect()
    # torch.cuda.empty_cache()

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    # if local_rank == 0:
    #     for file in Path(cfg.save_dir).glob("checkpoint-*"):
    #         for global_step_dir in file.glob("global_step*"):
    #             #delete the directory
    #             import shutil
    #             shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
