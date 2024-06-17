import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
import json
from utils import get_model_identifiers_from_yaml
import re
import random

def split_paragraph(paragraph):
    # 使用正则表达式将段落分割为句子
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

    # 初始化段落列表和当前段落
    paragraphs = []
    current_paragraph = ""

    # 遍历句子
    for sentence in sentences:
        # 将当前句子添加到当前段落
        current_paragraph += sentence

        # 如果当前段落的字数超过400，则添加到段落列表中，并开始下一个段落
        if len(current_paragraph) > 400:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""

    # 将最后一个段落添加到段落列表中（如果有的话）
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    return paragraphs

def convert_raw_data_to_model_format(tokenizer, max_length, text):

    full_text = text

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    # #change label to -100 for question tokens
    # for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class TextForgetDatasetWikipedia(Dataset):
    def __init__(self, data_path, tokenizer, content, random_content, model_family, max_length=512, split="wikipedia", loss_type="idk"):
        super(TextForgetDatasetWikipedia, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        paragraphs = split_paragraph(content)

        # 将句子列表写入一个JSON文件
        with open(data_path+'/sentences.json', 'w', encoding='utf-8') as f:
            json.dump(paragraphs, f, ensure_ascii=False)

        # 加载JSON文件作为数据集
        self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, 'sentences.json'))["train"]

        if random_content is not None:
            retain_text = ""
            for random_text in random_content:
                retain_text += random_text
            retain_text_paragraphs = split_paragraph(retain_text)

            # make sure the size of retain_text_paragraphs is larger than forgetting paragraphs
            while len(retain_text_paragraphs) < len(paragraphs):
                retain_text_paragraphs *= 2

            with open(data_path + '/retain_sentences.json', 'w', encoding='utf-8') as f:
                json.dump(retain_text_paragraphs, f, ensure_ascii=False)

            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, 'retain_sentences.json'))["train"]


        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        elif self.loss_type in ["grad_ascent"]:
            self.split1 = "forget"
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        # idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(
        #     self.retain_data)
        # question = data[idx]['question']
        # answer = data[idx]['answer']
        if self.loss_type in ["grad_ascent"]:
            text = self.forget_data[idx]['text']

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)
            rets.append(converted_data)
        else:
            for data_type in [self.split1, self.split2]:
                data = self.retain_data if data_type == "retain" else self.forget_data

                text = data[idx]['text']

                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)
                rets.append(converted_data)

        return rets



class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
