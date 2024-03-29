#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：03/08/2022 14:46 
# ====================================
import argparse
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.collate import gold_labels, collate_fn
from models.metrics import SpanEvaluator
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.processor import DataProcessor
from utils.tager import SpanLabel, RelationLabel

from tqdm import tqdm

import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(model, metric, data_loader, device, args):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            input_ids, attention_mask, words, spans, relations, span_labels, relation_labels, word_pos_to_token_pos, bio_labels = batch


            length = max(len(ids) for ids in input_ids)

            input_ids = torch.tensor([c + [0] * (length - len(c)) for c in input_ids], device=device)
            attention_mask = torch.tensor([c + [0] * (length - len(c)) for c in attention_mask], device=device)

            # forward
            bio_probability, spans_probability, span_indices, relations_probability, candidate_indices = model(
                input_ids, attention_mask, length, word_pos_to_token_pos)

            gold_span_labels = gold_labels(span_indices, spans, span_labels)
            gold_span_labels = torch.tensor(gold_span_labels).reshape(-1)
            gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
            gold_relation_labels = torch.tensor(gold_relation_labels).reshape(-1)
            
            if args.use_additional_head:
                bio_labels = torch.tensor(np.array(bio_labels)).reshape(-1)

            # print(bio_probability.shape, bio_labels.shape)
            # print(spans_probability.shape, gold_span_labels.shape, torch.tensor(spans).shape)
            # print(relations_probability.shape, gold_relation_labels.shape)


            
            num_correct, num_infer, num_label = metric.compute(relations_probability.cpu(),
                                                               gold_relation_labels,
                                                               relations)
            metric.update(num_correct, num_infer, num_label)


            num_correct, num_infer, num_label = metric.compute_ner(spans_probability.cpu(),
                                                               gold_span_labels,
                                                               gold_span_labels)
            metric.update_ner(num_correct, num_infer, num_label)


            if args.use_additional_head:
                
                num_correct, num_infer, num_label = metric.compute_bio(bio_probability.cpu(),
                                                                   bio_labels,
                                                                   bio_labels)
                metric.update_bio(num_correct, num_infer, num_label)
                
    precision, recall, f1 = metric.accumulate()
    precision_ner, recall_ner, f1_ner = metric.accumulate_ner()
    precision_bio, recall_bio, f1_bio = metric.accumulate_bio()
    model.train()

    if args.use_additional_head:
        return precision, recall, f1, precision_ner, recall_ner, f1_ner, precision_bio, recall_bio, f1_bio
    else:
        return precision, recall, f1, precision_ner, recall_ner, f1_ner

def do_eval():
    set_seed(1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"using device:{device}")
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # create processor
    processor = DataProcessor(tokenizer, args.max_seq_len)

    print("Loading Train & Eval Dataset...")
    # Load dataset
    test_dataset = CustomDataset("dev", args.test_path, processor, tokenizer, args.max_seq_len)

    print("Construct Dataloader...")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    # build span-aste model
    model = SpanAsteModel(
        args.bert_model,
        target_dim,
        relation_dim,
        device=device,
        span_maximum_length=args.span_maximum_length,
        use_additional_head=args.use_additional_head
    )

    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location=torch.device(device)))
    model.to(device)

    metric = SpanEvaluator()

    if args.use_additional_head:
        precision, recall, f1, precision_ner, recall_ner, f1_ner, precision_bio, recall_bio, f1_bio = evaluate(model, metric, test_dataloader, device, args)
        print("-----------------------------")
        print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
              (precision, recall, f1))
        print("Evaluation NER Precision: %.5f | Recall: %.5f | F1: %.5f" %
              (precision_ner, recall_ner, f1_ner))
        print("Evaluation BIO Precision: %.5f | Recall: %.5f | F1: %.5f" %
              (precision_bio, recall_bio, f1_bio))

    else:
        precision, recall, f1, precision_ner, recall_ner, f1_ner = evaluate(model, metric, test_dataloader, device, args)
        print("-----------------------------")
        print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
              (precision, recall, f1))
        print("Evaluation NER Precision: %.5f | Recall: %.5f | F1: %.5f" %
              (precision_ner, recall_ner, f1_ner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", type=str, default=None, help="The name of bert.")
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--load_args", type=str, default='True', help="Загрузить параметры из файла или нет")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--use_additional_head", default='False', choices=['False', 'True'], help="Use additional token classification head or not")
    parser.add_argument("--use_neutral_class", default='False', choices=['False', 'True'], help="Will there be a neutral class in the data")

    
    args = parser.parse_args()

    args.use_additional_head = {'False': False, 'True': True}[args.use_additional_head]
    args.use_neutral_class = {'False': False, 'True': True}[args.use_neutral_class]

    model_path = args.model_path
    test_path = args.test_path

    if args.load_args == 'True':
        with open('checkpoint/args.pkl', 'rb') as f:
            args = pickle.load(f)
    
        args.model_path = model_path
        args.test_path = test_path

    

    do_eval()
