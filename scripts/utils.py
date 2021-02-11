#!/usr/bin/env python
import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.tokens, self.labels = data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, args, label_alphabet):
        self.args = args
        self.label_alphabet = label_alphabet

        modes = 'train valid test'.split()
        self.data = []
        for mode in modes:
            path = os.path.join(args.data_dir, '{}.pkl'.format(mode))
            self.data.append(pkl.load(open(path, 'rb')))
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.cls, self.sep, self.pad = args.CLS, args.SEP, args.PAD
        #assert label_alphabet.get_index('O') == 0
        self.label_pad = 0

    def collate_fn(self, batch_data):
        input_ids, attention_masks, input_segments, input_labels, valid_ids, label_masks = zip(*batch_data)
        # print(input_ids)

        return {
            'input_ids': torch.tensor(input_ids).to(self.args.device),
            'attention_masks': torch.tensor(attention_masks).to(self.args.device),
            'input_segments': torch.tensor(input_segments).to(self.args.device),
            'input_labels': torch.tensor(input_labels).to(self.args.device),
            'valid_ids': torch.tensor(valid_ids).to(self.args.device),
            'label_masks': torch.tensor(label_masks).to(self.args.device)
        }
    
    def getdata(self, kfold=False):
        for data in self.data:
            yield DataLoader(MyDataset(data), shuffle=False, batch_size=self.args.batch_size, collate_fn=self.collate_fn)


def decode_from_path(path, label_map):
    # path: [[0, 1, 2], [1, 2, 3]]
    res, start = [], -1
    # path = [alhpabet.get_instance(w) for w in path]
    path = [label_map[w] for w in path]
    # path = list(map(lambda x:list(map(alhpabet.get_instance, x)), paths))
    for i, tag in enumerate(path):
        if tag == '[SEP]':
            break
        if tag.startswith('B'):
            if start != -1:
                res.append((start, i, path[start][2:]))
            start = i
        elif tag.startswith('I'):
            if start == -1:
                start = i
            elif path[start][1:] != tag[1:]:
                res.append((start, i, path[start][2:]))
                start = i
        else:
            if start != -1:
                res.append((start, i, path[start][2:]))
                start = -1
    if start != -1:
        res.append((start, i, path[start][2:]))
    return res
