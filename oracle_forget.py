from data_module import TextForgetDatasetWikipedia, TextForgetDatasetDPOQA, TextForgetDatasetKTOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from torch import nn
from transformers.trainer_utils import seed_worker
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import hydra
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml, set_random_seed
import json
import random

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

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
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    data_path = '/root/Unlearn_Harry_Potter/Baselines/negative-preference-optimization-main/TOFU/wikipedia_data'
    oracle_outputs_cache =  {}

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path,
                                                 use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model.eval()
    torch.no_grad()
    with open(data_path + "/llama_concepts_qa.json", "r", encoding="utf-8") as file:
        data = json.load(file)

        idy = 0
        for ix, item in enumerate(data):
            print('ix: ',ix)

            wikipedia_content = item['wikipedia_content']
            #random_wikipedia_content = random.sample([x['wikipedia_content'] for x in data if x != item], 3)

            torch_format_dataset = TextForgetDatasetWikipedia(data_path,
                                                              content=wikipedia_content,
                                                              random_content=None,
                                                              tokenizer=tokenizer,
                                                              model_family=cfg.model_family,
                                                              max_length=max_length,
                                                              split=cfg.split,
                                                              loss_type="grad_ascent")

            train_dataset = torch_format_dataset
            data_collator = custom_data_collator_forget

            dataloader_params = {
                "batch_size": 1,
                "collate_fn": data_collator,
                # "num_workers": self.args.dataloader_num_workers,
                # "pin_memory": self.args.dataloader_pin_memory,
                # "persistent_workers": self.args.dataloader_persistent_workers,
            }

            generator = torch.Generator()
            generator.manual_seed(cfg.seed)
            # print(f'Generator........Epoch-{self.state.global_step}')

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["generator"] = generator
                dataloader_params["shuffle"] = True  # set shuffle=True with specified generator.
                # dataloader_params["sampler"] = self._get_train_sampler()
                # dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker

            dataloader = DataLoader(train_dataset, **dataloader_params)

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

            for batch_idx, inputs in enumerate(dataloader):
                # print(f"Batch Index: {batch_idx}")
                input_ids, labels, attention_mask = inputs[0]
                input_ids, labels, attention_mask = input_ids.cuda(), labels.cuda(), attention_mask.cuda()
                assert len(input_ids) == 1
                oracle_outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                #print('oracle_outputs: ',oracle_outputs)
                oracle_loss = get_batch_loss(oracle_outputs.logits, labels)
                #print('oracle_loss: ',oracle_loss)

                oracle_outputs_cache[str(list(input_ids.squeeze(0).detach().cpu().numpy()))]=oracle_loss.detach().cpu()
            #print('oracle_outputs_cache: ',oracle_outputs_cache)
                # idy+=1
                # if idy % 50 == 0:
                #     print(f'saving on {idy} step.')
                #     torch.save(oracle_outputs_cache, f'/root/autodl-tmp/datasets/oracle_outputs_cache_loss{idy}.pt')
                #     oracle_outputs_cache = {}
            torch.save(oracle_outputs_cache, f'/root/autodl-tmp/datasets/oracle_outputs_cache_loss_.pt')


if __name__ == "__main__":
    main()