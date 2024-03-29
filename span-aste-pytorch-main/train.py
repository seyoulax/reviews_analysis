#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/18 16:19
@Desc    ：
==================================================
"""
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
# from models.losses import log_likelihood
from models.metrics import SpanEvaluator
from utils.bar import ProgressBar
from utils.dataset import CustomDataset
from models.collate import collate_fn, gold_labels
import numpy as np
from models.model import SpanAsteModel
from utils.processor import DataProcessor
from utils.tager import SpanLabel
from utils.tager import RelationLabel
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm

import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"using device:{device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def do_train():
    # set seed
    set_seed(args.seed)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # create processor
    processor = DataProcessor(tokenizer, args.max_seq_len, use_neutral_class=args.use_neutral_class)

    print("Loading Train & Eval Dataset...")
    # Load dataset
    train_dataset = CustomDataset("train", args.train_path, processor, tokenizer, args.max_seq_len)
    eval_dataset = CustomDataset("dev", args.dev_path, processor, tokenizer, args.max_seq_len)

    print("Construct Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
    model.to(device)

    if args.init_from_ckpt is not None:
        model.load_state_dict(torch.load(args.init_from_ckpt))

    bert_param_optimizer = list(model.bert.named_parameters()) # Параметры Берта, ему меньший learning rate потом дадим
    other_params = []
    for name, _ in model.named_children():
        if name == 'bert': continue
        other_params.extend(list(model.get_submodule(name).named_parameters())) # Всё остальное


    optimizer_grouped_parameters = [

        {'params': [p for n, p in bert_param_optimizer],
        'weight_decay': args.weight_decay_bert, 'lr': args.learning_rate_bert},

        {'params': [p for n, p in other_params],
         'weight_decay': args.weight_decay_other, 'lr': args.learning_rate_other},

    ]

    print("Building Optimizer...")
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay_other, lr=args.learning_rate_other)
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    metric = SpanEvaluator()

    tic_train = time.time()
    global_step = 0
    best_f1 = 0
    loss_list = []
    num_batches = len(train_dataloader)
    logging_steps = np.linspace(1, num_batches, args.logging_steps+1).astype(int)[1:]
    valid_steps = np.linspace(1, num_batches, args.valid_steps+1).astype(int)[1:]
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    for epoch in range(1, args.num_epochs + 1):
        pbar_custom = ProgressBar(n_total=num_batches, desc='Training')
        model.train()
        relative_step = 0
    
        for batch_ix, batch in (pbar := tqdm(enumerate(train_dataloader), total=num_batches)):
            input_ids, attention_mask, words, spans, relations, span_labels, relation_labels, word_pos_to_token_pos, bio_labels = batch
            

            length = max(len(ids) for ids in input_ids)
            input_ids = torch.tensor([c + [0] * (length - len(c)) for c in input_ids], device=device)
            attention_mask = torch.tensor([c + [0] * (length - len(c)) for c in attention_mask], device=device)

            
            bio_probability, spans_probability, span_indices, relations_probability, candidate_indices = model(
                input_ids, attention_mask, length, word_pos_to_token_pos)
            
            gold_span_labels = gold_labels(span_indices, spans, span_labels) # Удобная форма для таргетов для классификации спанов
            gold_span_labels = torch.tensor(gold_span_labels, device=device).reshape(-1)

            batch_size = spans_probability.shape[0]
            
            if batch_size != 1:
              spans_probability = torch.reshape(spans_probability, (batch_size * spans_probability.shape[1], spans_probability.shape[2]))
            else:
              spans_probability = spans_probability[0]
            
            if args.use_additional_head:
                bio_labels = torch.tensor(np.array(bio_labels), device=device)
                loss_bio = criterion(bio_probability.permute(0, 2, 1), bio_labels)
            

            loss_ner = criterion(spans_probability, gold_span_labels) # Считаем лосс на классификации спанов


            gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels) # Удобная форма для таргетов для классификации пар аспект-мнение
            gold_relation_labels = torch.tensor(gold_relation_labels, device=device).reshape(-1)
            

            if batch_size != 1:
                
              relations_probability = torch.reshape(relations_probability, (batch_size * relations_probability.shape[1], relations_probability.shape[2]))
            else:
              relations_probability = relations_probability[0]
            
            loss_relation = criterion(relations_probability, gold_relation_labels) # Считаем лосс на классификации пар аспект-мнение

            if args.use_additional_head:
                loss = (args.bio_weight * loss_bio + args.ner_weight * loss_ner + args.relation_weight * loss_relation) / gradient_accumulation_steps # Суммируем лосс
            else:
                loss = (args.ner_weight * loss_ner + args.relation_weight * loss_relation) / gradient_accumulation_steps # Суммируем лосс

            loss.backward()

            if (relative_step + 1) % gradient_accumulation_steps == 0 or relative_step ==  num_batches -1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            
            

            loss_list.append(float(loss.item()))
            # info, show_bar = pbar_custom(batch_ix, {"loss": float(loss.item())})
            # # if batch_ix % 2 ==0 :
            # if len(info) != 0:
            #     show_info = f'{show_bar} ' + \
            #                 "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            #     # print(show_info, end='')
            #     pbar.set_description(show_info)
            # else:
            #     pbar.set_description(show_bar)
            pbar.set_description(f'loss: {loss.item()}')
                
            global_step += 1
            relative_step += 1
            if relative_step in logging_steps or relative_step == num_batches:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)

                print(
                    "\nglobal step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg,
                       args.logging_steps / time_diff))
                tic_train = time.time()

            # valid
            if relative_step in valid_steps or relative_step == num_batches:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                

                if args.use_additional_head:
                    precision, recall, f1, precision_ner, recall_ner, f1_ner, precision_bio, recall_bio, f1_bio = evaluate(model, metric, eval_dataloader, device, args)
                    print("-----------------------------")
                    print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                          (precision, recall, f1))
                    print("Evaluation NER Precision: %.5f | Recall: %.5f | F1: %.5f" %
                          (precision_ner, recall_ner, f1_ner))
                    print("Evaluation BIO Precision: %.5f | Recall: %.5f | F1: %.5f" %
                          (precision_bio, recall_bio, f1_bio))
            
                else:
                    precision, recall, f1, precision_ner, recall_ner, f1_ner = evaluate(model, metric, eval_dataloader, device, args)
                    print("-----------------------------")
                    print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                          (precision, recall, f1))
                    print("Evaluation NER Precision: %.5f | Recall: %.5f | F1: %.5f" %
                          (precision_ner, recall_ner, f1_ner))

                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

                tic_train = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Версия Bert")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    
    parser.add_argument("--learning_rate_bert", default=5e-5, type=float, help="The initial learning rate for BERT.")
    parser.add_argument("--learning_rate_other", default=1e-3, type=float, help="The initial learning rate for other params.")
    parser.add_argument("--weight_decay_bert", default=1e-2, type=float, help="The initial learning rate for BERT.")
    parser.add_argument("--weight_decay_other", default=0, type=float, help="The initial learning rate for other params.")

    parser.add_argument("--span_maximum_length", default=8, type=int, help="The maximum length of span to be considered as an aspect/opinion")
    parser.add_argument("--use_neutral_class", default='False', choices=['False', 'True'], help="Will there be a neutral class in the data")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--train_path", default="data/15res", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="data/15res", type=str, help="The path of dev set.")
    
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=1, type=int, help="Logging steps per epoch")
    parser.add_argument("--valid_steps", default=1, type=int,
                        help="Evaluation steps per epoch")
    
    parser.add_argument("--init_from_ckpt", default=None, type=str,
                        help="The path of model parameters for initialization.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps during training")
    parser.add_argument("--use_additional_head", default='False', choices=['False', 'True'], help="Use additional token classification head or not")

    parser.add_argument("--ner_weight", default=1, type=float, help="Loss weight for NER")
    parser.add_argument("--relation_weight", default=1, type=float, help="Loss weight for relation (aspect+opinion pair classification)")
    parser.add_argument("--bio_weight", default=1, type=float, help="Loss weight for additional BIO classification head")

    args = parser.parse_args()

    args.use_neutral_class = {'False': False, 'True': True}[args.use_neutral_class]
    args.use_additional_head = {'False': False, 'True': True}[args.use_additional_head]

    with open('checkpoint/args.pkl', 'wb') as f:
        pickle.dump(args, f)


    do_train()
