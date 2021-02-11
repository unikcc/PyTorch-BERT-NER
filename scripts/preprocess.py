#!/usr/bin/env python

import tqdm
import json
import os
import pickle as pkl
from transformers import BertTokenizer
import yaml
from attrdict import AttrDict
from alphabet import Alphabet
import numpy as np


class Preprocessor:
    """
    my class for preprocess data
    """
    def __init__(self):
        config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        config = AttrDict(config)
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=False)
        if not os.path.exists(self.config.data_dir):
            os.makedirs(self.config.data_dir)
        self.CLS, self.SEP = config.CLS, config.SEP
        self.label_alphabet = Alphabet('label', padflag=False, unkflag=False, init_list=['O', self.CLS, self.SEP])

    def read_file(self, filename):
        f = open(filename)
        data, sentence, label= [], [], []
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    sentence, label = [], []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])
        if len(sentence) >0:
            data.append((sentence,label))
        return data

    def get_label(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    
    def transform2indices(self, data, mode='train'):
        max_seq_length = self.config.max_seq_length
        label_list = self.get_label()
        label_map = {label : i for i, label in enumerate(label_list)}
        res = []
        for sentence, label_list in data:
            labels = label_list
            tokens, valid_ids = [], []
            for i, word in enumerate(sentence):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                valid_ids += [1] + [0 for w in range(len(token) - 1)]
            label_mask = [1] * len(labels)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[:max_seq_length - 2]
                labels = labels[:max_seq_length - 2]
                valid_ids = valid_ids[:max_seq_length - 2]
                label_mask = label_mask[:max_seq_length - 2]

            tokens = [self.CLS] + tokens + [self.SEP]
            labels = [self.CLS] + labels + [self.SEP]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_labels = [label_map[w] for w in labels]
            valid_ids = [1] + valid_ids + [1]
            label_mask = [1] + label_mask + [1]

            valid_ids = np.where(np.array(valid_ids) == 1)[0].tolist()

            input_ids += [0] * (max_seq_length - len(input_ids))
            input_mask += [0] * (max_seq_length - len(input_mask))
            input_labels += [0] * (max_seq_length - len(input_labels))
            valid_ids += [0] * (max_seq_length - len(valid_ids))
            label_mask += [0] * (max_seq_length - len(label_mask))
            input_segments = [0] * len(input_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(input_segments) == max_seq_length
            assert len(input_labels) == max_seq_length
            assert len(valid_ids) == max_seq_length
            assert len(label_mask) == max_seq_length
            res.append((input_ids, input_mask, input_segments, input_labels, valid_ids, label_mask))
        path = os.path.join(self.config.data_dir, '{}.pkl'.format(mode))
        pkl.dump(res, open(path, 'wb'))

    def manage(self):
        modes = ['train', 'valid', 'test']
        dataset = []
        for mode  in modes:
            filename = os.path.join(self.config['dataset_dir'], '{}.txt'.format(mode))
            print("Start preprocess {}".format(mode))
            dataset.append(self.read_file(filename))
            print("End preprocess {}".format(mode))
        for mode, data in zip(modes, dataset):
            self.transform2indices(data, mode)

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.manage()