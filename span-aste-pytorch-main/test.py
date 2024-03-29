#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：05/08/2022 9:57 
# ====================================
import os

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.collate import collate_fn
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.tager import SpanLabel, RelationLabel
from main import countfiles

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="banki", type=str)
    parser.add_argument(
        "--bert_model",
        default="ai-forever/ruBert-base",
        type=str,
        help="Path to pre-trained Bert model or shortcut name",
    )

    parser.add_argument(
        "--path_to_checkpoint",
        default="checkpoint/model_best/model.pt",
        type=str,
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--path_to_dataset",
        default="data/banki/test_triplets.txt",
        type=str,
        help="Path to data",
    )

    parser.add_argument("--max_length", default=512, type=int)

    parser.add_argument(
        "--type", default='test', choices=['test', 'inference'], help="Проводится оценка на тестовом датасете или инференс"
    )

    parser.add_argument(
        "--verbose", default='False', choices=['False', 'True']
    )

    parser.add_argument("--use_additional_head", default='False', choices=['False', 'True'], help="Use additional token classification head or not")
    parser.add_argument("--use_neutral_class", default='False', choices=['False', 'True'], help="Will there be a neutral class in the data")
    parser.add_argument("--span_maximum_length", default=8, type=int, help="The maximum length of span to be considered as an aspect/opinion")

    args = parser.parse_args()

    args.use_additional_head = {'False': False, 'True': True}[args.use_additional_head]
    args.use_neutral_class = {'False': False, 'True': True}[args.use_neutral_class]

    return args


def compute_f1(labels, preds):
  n_tp, n_pred, n_gold = 0, 0, 0
  labels = labels.values
  preds = preds.values

  for label, pred in zip(labels, preds):
    n_pred += len(pred)
    n_gold += len(label)

    for triplet in label:
      if triplet in pred:
        n_tp += 1

  # print(n_tp, n_pred, n_gold)

  precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
  recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
  f1 = (
      2 * precision * recall / (precision + recall)
      if precision != 0 or recall != 0
      else 0
  )
  scores = {"precision": precision, "recall": recall, "f1": f1}

  return scores
  



args = init_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"using device:{device}")
# tokenizer
tokenizer = BertTokenizer.from_pretrained(args.bert_model)

print("Building SPAN-ASTE model...")
# get dimension of target and relation
target_dim, relation_dim = len(SpanLabel), len(RelationLabel)

if args.use_neutral_class == False:
    relation_dim -= 1

# build span-aste model
model = SpanAsteModel(
    args.bert_model,
    target_dim,
    relation_dim,
    device=device,
    use_additional_head=args.use_additional_head,
    span_maximum_length=args.span_maximum_length
)

model.load_state_dict(torch.load(args.path_to_checkpoint, map_location=torch.device(device)))
model.to(device)
model.eval()

with open(args.path_to_dataset, "r", encoding="utf8") as f:
    data = f.readlines()
res = []
for d in tqdm(data):
    text, label = d.strip().split("####")
    words = text.split()

    # tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    # input = tokenizer(text, max_length=args.max_length, truncation=True, return_tensors="pt")

    # input_ids = input.input_ids
    # attention_mask = input.attention_mask
    # token_type_ids = input.token_type_ids
    # length = input_ids.shape[0]

    input_ids = [101]

    word_pos_to_token_pos = dict()
    
    count_ids = 1

    for i, word in enumerate(words): # Идем по словам, каждое переводим в индексы токенов, добавляем в общий список токенов
        ids_word  = tokenizer.encode(word, add_special_tokens=False)
        input_ids.extend(ids_word)

        word_pos_to_token_pos[i] = (count_ids, count_ids + len(ids_word))
        count_ids += len(ids_word)

        if count_ids >= args.max_length - 1: # Вышли за ограничение по токенам. -1 ставим для end token (102).

            if word_pos_to_token_pos[i][1] > args.max_length - 1: # Если токены из последнего токена выходят за ограничение, последнее слово удаляем
                del word_pos_to_token_pos[i]
            
            break


    input_ids = input_ids[:args.max_length - 1]
    input_ids.append(102) # End token.   [101, 125, 532, 65433, 12, 102]
    
    attention_mask = torch.tensor([1] * len(input_ids), device=device)
    length = len(input_ids)

    # forward
    bio_probability, spans_probability, span_indices, relations_probability, candidate_indices = model(
                torch.tensor(input_ids, device=device).reshape(1, -1), attention_mask.reshape(1, -1), 
                length, (word_pos_to_token_pos, ))

    relations_probability = relations_probability.squeeze(0)
    predict = []
    predict_text = []
    for idx, can in enumerate(candidate_indices[0]):
        a, b, c, d = can
        aspect = tokenizer.convert_tokens_to_string(words[a:b])
        opinion = tokenizer.convert_tokens_to_string(words[c:d])
        sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name

        if sentiment != RelationLabel.INVALID.name:
            predict_text.append((aspect, opinion, sentiment))
            predict.append((list(range(a, b)), list(range(c, d)), sentiment))

    if args.verbose == 'True':
      print("text:", text)
      print("predict", predict_text)


    if args.type == 'test':
      labels = []
      labels_text = []

      for l in eval(label):
          a, o, sm = l
          labels.append((a, o, sm))
          a = " ".join([words[i] for i in a])
          o = " ".join([words[i] for i in o])
          labels_text.append((a, o, sm))
      if args.verbose == 'True':
        print("label", labels_text)
      res.append({"text": text, "pred": predict, "label": labels, "pred_text": predict_text, "label_text": labels_text})
    
    else:
      res.append({"text": text, "pred": predict, "pred_text": predict_text})

df = pd.DataFrame(res)
df.to_csv(f"results/results{countfiles}.csv", index=False)
print("results/results{countfiles}.csv")
if args.type == 'test':
    print(compute_f1(df['label'], df['pred']))
