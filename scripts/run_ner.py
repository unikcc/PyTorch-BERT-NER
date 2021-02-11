# from __future__ import absolute_import, division, print_function

import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from torch import nn
from tqdm import tqdm, trange

from model import Ner, myClassification
from metric import Metric
import yaml
from attrdict import AttrDict
from utils import MyDataLoader, decode_from_path
from alphabet import Alphabet
from seqeval.metrics import classification_report

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Main:
    def __init__(self):
        config = AttrDict(yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader))
        self.config = config
        self.config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.label_alphabet = Alphabet('label')
        # self.label_alphabet.load(config.data_dir, 'label')
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = MyDataLoader(config, self.label_alphabet).getdata()

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if os.path.exists(config.output_dir) and len(os.listdir(config.output_dir)) > 0:
            raise FileExistsError("Output dir: {} has been existed".format(config.output_dir))
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        self.label_list = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def train(self):
        label_map = {i : label for i, label in enumerate(self.label_list)}
        global_step = 0
        self.model.train()
        for _ in trange(self.config.epoch_size, desc="Epoch"):
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch.values()
                loss = self.model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        model_to_save.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        label_map = {i : label for i, label in enumerate(self.label_list)}
        model_config = {"bert_model":self.config.bert_path,"do_lower":self.config.do_lower_case,"max_seq_length":self.config.max_seq_length,"num_labels":len(self.label_list),"label_map":label_map}
        json.dump(model_config,open(os.path.join(self.config.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    
    def eval(self):
        self.metric = Metric()
        self.model.eval()
        y_true, y_pred = [], []
        label_map = {i : label for i, label in enumerate(self.label_list)}
        for batch in tqdm(self.valid_dataloader, desc="Evaluating"):
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch.values()
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
            logits = torch.argmax(logits, -1).tolist()
            label_ids = label_ids.tolist()
            for i in range(len(label_ids)):
                pred = decode_from_path(logits[i], label_map)
                gold = decode_from_path(label_ids[i], label_map)
                self.metric.add_instance(pred, gold)
        p, r, f, report = self.metric.report()

        logger.info("\n%s", report)
        output_eval_file = os.path.join(self.config.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
        
    def forward(self):
        config = self.config
        num_labels = len(self.label_list)

        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=config.do_lower_case)

        num_train_optimization_steps = self.train_dataloader.__len__() / config.gradient_accumulation_steps * config.epoch_size

        bert_config = BertConfig.from_pretrained(config.bert_path, num_labels=num_labels)

        self.model = myClassification.from_pretrained(config.bert_path, config = bert_config).to(device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        warmup_steps = int(config.warmup_proportion * num_train_optimization_steps)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=float(config.adam_epsilon))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

        self.train()
        self.eval()

if __name__ == "__main__":
    main = Main()
    main.forward()
