#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/22 10:09 
@Desc    ：
==================================================
"""
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import LSTM, init
import itertools
import pickle
import time

from transformers import BertModel

from utils.tager import SpanLabel


class SpanRepresentation(nn.Module):
    """
    We define each span representation si,j ∈ S as:
            si,j =   [hi; hj ; f_width(i, j)] if BiLSTM
                     [xi; xj ; f_width(i, j)] if BERT
    where f_width(i, j) produces a trainable feature embedding representing the span width (i.e., j −i+ 1)
    Besides the concatenation of the start token, end token, and width representations,the span representation si,j
    can also be formed by max-pooling or mean-pooling across all token representations of the span from position i to j.
    The experimental results can be found in the ablation study.
    """

    def __init__(self, span_width_embedding_dim, span_maximum_length):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = span_maximum_length
        self.span_width_embedding = nn.Embedding(span_maximum_length, span_width_embedding_dim) # Эмбеддинг длины спана 

    def span_length_embedding(self, width, device): # Эмбеддинг длины спана 
        return self.span_width_embedding(torch.LongTensor([width-1]).to(device))

    def forward(self, x: Tensor, batch_max_seq_len):
        """
        Формируем эмбеддинги спанов
        
        [[2, 5], [0, 1], [1, 2], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
        :param x: batch * len * dim
        :param term_cat:
        :return:
        """
        batch_size, sequence_length, _ = x.size()
        device = x.device

        len_arrange = torch.arange(0, batch_max_seq_len, device=device)
        span_indices = []

        spans = []

        x = x.permute(1, 0, 2) # Длина последовательности X batch_size X 768

        max_window = min(batch_max_seq_len, self.span_maximum_length) # Чтобы не брать спаны длины 5, когда длина предложения 3 слова

        for window in range(1, max_window + 1):
            span_length_embedding = self.span_length_embedding(window, device).repeat((batch_size, 1)) # Эмбеддинг для такой длины спана
            if window == 1:
                indics = [(x.item(), x.item()) for x in len_arrange] # Индексы спанов
            else:
                res = len_arrange.unfold(0, window, 1)
                indics = [(idx[0].item(), idx[-1].item()) for idx in res] # Индексы спанов: [(0, 1), (1, 2)]
            for s in indics:
              spans.append(torch.cat(
                      (x[s[0]], x[s[1]], span_length_embedding), dim=1
                      )) # Берем 2 эмбеддинга (первый и последний токен в спане), прибавляем к ним эмбеддинг длины спана
            span_indices.extend(indics)
            

        return torch.stack(spans, dim=1), span_indices


class PrunedTargetOpinion:
    """
    For a sentence X
    of length n, the number of enumerated spans is O(n^2), while the number of possible pairs between
    all opinion and target candidate spans is O(n^4) at the later stage (i.e., the triplet module). As such,
    it is not computationally practical to consider all possible pairwise interactions when using a span-based
    approach. Previous works (Luan et al., 2019; Wadden  et al., 2019) employ a pruning strategy to
    reduce the number of spans, but they only prune the spans to a single pool which is a mix of different
    mention types. This strategy does not fully consider
    """

    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        target_indices = torch.topk(spans_probability[:, :, SpanLabel.ASPECT.value], nz, dim=-1).indices # Берем самые вероятные аспекты
        opinion_indices = torch.topk(spans_probability[:, :, SpanLabel.OPINION.value], nz, dim=-1).indices # Берем самые вероятные мнения
        return target_indices, opinion_indices


class TargetOpinionPairRepresentation(nn.Module):
    """
    Target Opinion Pair Representation We obtain the target-opinion pair representation by coupling each target candidate representation
    St_a,b ∈ St with each opinion candidate representation So_a,b ∈ So:
        G(St_a,b,So_c,d) = [St_a,b; So_c,d; f_distance(a, b, c, d)] (5)
    where f_distance(a, b, c, d) produces a trainable feature embedding based on the distance (i.e., min(|b − c|, |a − d|)) between the target
    and opinion span
    """

    def __init__(self, distance_embeddings_dim):
        super(TargetOpinionPairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 8, 16, 32, 64] # Бакеты (для расстояний 5, 6, 7 один и тот же эмбеддинг и т.д.)
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), distance_embeddings_dim) # Эмбеддинг расстояния между аспектом и мнением

    def min_distance(self, a, b, c, d):
        return min(abs(b - c), abs(a - d))

    def bucket_embedding(self, width, device): # Эмбеддинг расстояния между аспектом и мнением
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.distance_embeddings(torch.LongTensor([em]).to(device))

    def forward(self, spans, span_indices, target_indices, opinion_indices):
        """
        Формируем эмбеддинги потенциальных пар аспект/мнение

        :param spans:
        :param span_indices:
        :param target_indices:
        :type
        :param opinion_indices:
        :return:
            candidate_indices :
                List[List[Tuple(a,b,c,d)]]
            relation_indices :
                List[List[Tuple(span1,span2)]]
        """

      
        batch_size = spans.size(0)
        device = spans.device

        candidate_indices, relation_indices = [], []
        
        for batch_idx in range(batch_size):
            # Формируем индексы всех возможных пар аспект/мнение: [0, 4, 7] и [2, 3, 6] превращаем в [(0, 2), (0, 3), (0, 6), (4, 2), (4, 3), (4, 6), (0, 2), (4, 3), (7, 6)]
            pairs = list(itertools.product(target_indices[batch_idx].cpu().tolist(), opinion_indices[batch_idx].cpu().tolist()))
            
            relation_indices.append(pairs)
            candidate_ind = []
            for pair in pairs:
                a, b = span_indices[pair[0]]
                c, d = span_indices[pair[1]]
                candidate_ind.append((a, b, c, d))
            candidate_indices.append(candidate_ind)
            
        min_distances_all = []
        for batch_idx in range(batch_size):
            min_distances_all.extend(self.min_distance(*span_indices[c[0]], *span_indices[c[1]]) for c in relation_indices[batch_idx])
        min_distances_unique = np.unique(min_distances_all)
        min_distance_to_bucket_embedding = dict(zip(min_distances_unique, [self.bucket_embedding(min_distance, device).squeeze(0) for min_distance in min_distances_unique]))
        # Формируем словарь вида {расстояния между аспектом и мнением : номер эмбеддинга (бакета)}

        
        candidate_pool = []

        
        for batch_idx in range(batch_size):
                
            
            min_distances = [self.min_distance(*span_indices[c[0]], *span_indices[c[1]]) for c in relation_indices[batch_idx]]
            

            bucket_embeddings = torch.stack([min_distance_to_bucket_embedding[min_distance] for min_distance in min_distances]) # Эмбеддинги расстояний для каждой пары аспект-мнение 

            target_spans = spans[batch_idx, target_indices[batch_idx], :] # Эмбеддинги всех потенциальных аспектов (срез быстрее происходит, чем из всех спанов каждый раз)
            opinion_spans = spans[batch_idx, opinion_indices[batch_idx], :] # Эмбеддинги всех потенциальных мнений (срез быстрее происходит, чем из всех спанов каждый раз)

            shape1 = target_spans.shape
            shape2 = opinion_spans.shape

            # Декартово произведение, но для строк тензоров
            target_spans = target_spans.unsqueeze(1).expand(-1, shape2[0], -1)
            opinion_spans = opinion_spans.unsqueeze(0).expand(shape1[0], -1, -1)

            result_tensor = torch.cat((target_spans, opinion_spans), dim=2)
            result_tensor = result_tensor.view(-1, shape1[1]*2)
           
            
            candidate_pool.append(torch.cat([result_tensor, bucket_embeddings], dim=1))
            

        return torch.stack(candidate_pool), candidate_indices, relation_indices


class SpanAsteModel(nn.Module):
    """
    This repository is a pytorch version that implements Ali's ACL 2021 research paper
    `Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction`
    paper:https://aclanthology.org/2021.acl-long.367/
    """

    def __init__(
            self,
            pretrain_model,
            target_dim: "int",
            relation_dim: "int",
            ffnn_hidden_dim: "int" = 150,
            span_width_embedding_dim: "int" = 20,
            span_maximum_length: "int" = 8,
            span_pruned_threshold: "int" = 0.5,
            pair_distance_embeddings_dim: "int" = 128,
            device="cpu",
            use_additional_head=True
    ) -> None:
        """
        :param input_dim: The number of expected features in the input `x`.
        :type int
        :param target_dim: The number of expected features for target .
        :type int
        :param relation_dim: The number of expected features for pairs .
        :type int
        :param lstm_layer: Number of lstm layers.
        :type int (default:1)
        :param lstm_hidden_dim: The number of features in the lstm hidden state `h`.
        :type int (default:1)
        :param lstm_bidirectional:
        :type boolean (default:300)
        :param ffnn_hidden_dim: The number of features in the feedforward hidden state `h`.
        :type int (default:150)
        :param span_width_embedding_dim: The number of features in the span width embedding layer.
        :type int (default:20)
        :param span_maximum_length: The maximum span length.
        :type int (default:8)
        :param span_pruned_threshold: threshold hyper-parameter for span pruned.
        :type int (default:0.5)
        :param pair_distance_embeddings_dim: The number of features in the target-opinion pair distance embedding layer.
        :type int (default:128)
        """
        super(SpanAsteModel, self).__init__()
        self.span_pruned_threshold = span_pruned_threshold
        self.pretrain_model = pretrain_model
        self.device = device
        self.use_additional_head = use_additional_head

        self.bert = BertModel.from_pretrained(pretrain_model)
        encoding_dim = self.bert.config.hidden_size

        self.span_representation = SpanRepresentation(span_width_embedding_dim, span_maximum_length)
        span_dim = encoding_dim * 2 + span_width_embedding_dim
        self.span_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, target_dim, bias=True)
        )
        self.pruned_target_opinion = PrunedTargetOpinion()
        self.target_opinion_pair_representation = TargetOpinionPairRepresentation(pair_distance_embeddings_dim)
        pairs_dim = 2 * span_dim + pair_distance_embeddings_dim
        self.pairs_ffnn = torch.nn.Sequential(
            nn.Linear(pairs_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, relation_dim, bias=True)
        )

        if self.use_additional_head: 
            self.bio_classification = torch.nn.Sequential(
                nn.Linear(encoding_dim, ffnn_hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(ffnn_hidden_dim, 3, bias=True)
            )
        
        self.reset_parameters()

    def reset_parameters(self):
        for name, _ in self.named_children():
            if name == 'bert': continue

            for n, p in self.get_submodule(name).named_parameters():
                if 'weight' in n:
                    init.xavier_normal_(p)

    def forward(self, input_ids, attention_mask, seq_len, word_pos_to_token_pos):
        """
        :param x: B * L * D
        :param adj: B * L * L
        :return:
        """
        

        batch_size, sequence_len = input_ids.size()




        
        # start_time = time.time()
        # x = self.bert(input_ids, attention_mask).last_hidden_state # Получаем эмбеддинги токенов с помощью BERT
        # cur_time = time.time()
        # print('\n', cur_time - start_time)
        # start_time = cur_time
        # for i in range(batch_size):
        #     x = self.bert(input_ids[[i]], attention_mask[[i]]).last_hidden_state # Получаем эмбеддинги токенов с помощью BERT
        # cur_time = time.time()
        # print(cur_time - start_time)



        x = self.bert(input_ids, attention_mask).last_hidden_state # Получаем эмбеддинги токенов с помощью BERT
        
        batch_embeddings = []
        embedding_max_length = max(len(di.keys()) for di in word_pos_to_token_pos)
        embeddings_lengths = []

        for batch_idx in range(x.shape[0]):
          embeddings = []
          for word, positions in word_pos_to_token_pos[batch_idx].items():
            embedding = torch.mean(x[batch_idx, positions[0] :positions[-1]], dim=0) # Усредняем эмбеддинги токенов, из которых состоит слово
            embeddings.append(embedding)
          embedding_max_length = max(embedding_max_length, len(embeddings))
          embeddings_lengths.append(embeddings)
            
          embeddings = torch.stack(embeddings)
          embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, embedding_max_length - len(embeddings)))
          batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        if self.use_additional_head:
            bio_probability = self.bio_classification(batch_embeddings)
        else:
            bio_probability = None
        
        spans, span_indices = self.span_representation(batch_embeddings, embedding_max_length)

    
        spans_probability = self.span_ffnn(spans) # Классификация спанов
        
        nz = int(embedding_max_length * self.span_pruned_threshold) # Сколько аспектов/мнений берем
        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz) # Потенциальные аспекты/мнения
        
        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, target_indices, opinion_indices) # Формируем все возможные пары аспект-мнение

        
        candidate_probability = self.pairs_ffnn(candidates) # Классифицируем пары аспект-мнение

        
        span_indices = [span_indices] * batch_size

        return bio_probability, spans_probability, span_indices, candidate_probability, candidate_indices
