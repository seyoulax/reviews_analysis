#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：03/08/2022 9:14 
# ====================================

import copy
import json
import os

from utils.tager import SentimentTriple, SentenceTagger


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, input_ids, text_a, words, spans, span_labels, relations, relation_labels, word_pos_to_token_pos, bio_labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.input_ids = input_ids
        self.text_a = text_a
        self.words = words
        self.spans = spans
        self.relations = relations
        self.span_labels = span_labels
        self.relation_labels = relation_labels
        self.word_pos_to_token_pos = word_pos_to_token_pos
        self.bio_labels = bio_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DataProcessor(object):
    def __init__(self, tokenizer, max_length, use_neutral_class=False):
        self.tokenizer = tokenizer
        self.max_length = max_length  # Сколько токенов от отзыва оставляем
        self.use_neutral_class = use_neutral_class # Есть ли нейтральный класс в данных

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(data_dir), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(data_dir), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line["text"]
            words = text.split()
            labels = line["labels"]
            

            input_ids = [101] # Start token

            word_pos_to_token_pos = dict() # Показывает, каким токенам соответствует i-e слово.  {0: (0, 1), 1: (1, 3)}. Нужно, чтобы потом эмбеддинги токенов одного слова усреднить.

            count_ids = 1 # Сколько уже токенов

            for i, word in enumerate(words): # Идем по словам, каждое переводим в индексы токенов, добавляем в общий список токенов
              ids_word  = self.tokenizer.encode(word, add_special_tokens=False)
              input_ids.extend(ids_word)

              word_pos_to_token_pos[i] = (count_ids, count_ids + len(ids_word))
              count_ids += len(ids_word)

              if count_ids >= self.max_length - 1: # Вышли за ограничение по токенам. -1 ставим для end token (102).

                if word_pos_to_token_pos[i][1] > self.max_length - 1: # Если токены из последнего токена выходят за ограничение, последнее слово удаляем
                  del word_pos_to_token_pos[i]
                
                break


            input_ids = input_ids[:self.max_length - 1]
            input_ids.append(102) # End token.   [101, 125, 532, 65433, 12, 102]


            sentiment_triples = [] # Триплеты
            for label in labels:
                aspect, opinion, sentiment = label
                if len(aspect) == 1:
                    aspect = aspect * 2 # [1] -> [1, 1]
                elif len(aspect) >= 2:
                    aspect = [aspect[0], aspect[-1]]  # [0, 1, 2] -> [0, 2]
                  
                if len(opinion) == 1:
                    opinion = opinion * 2 # [1] -> [1, 1]
                elif len(opinion) >= 2:
                    opinion = [opinion[0], opinion[-1]]  # [0, 1, 2] -> [0, 2]
                
                a1, a2 = aspect
                o1, o2 = opinion


                sentiment_triple = SentimentTriple.from_sentiment_triple(
                    ([a1, a2+1], [o1, o2+1], sentiment)) # Просто удобная форма представления данных

                sentiment_triples.append(sentiment_triple)

            sentence_tagger = SentenceTagger(sentiment_triples, use_neutral_class=self.use_neutral_class, length=len(words))
            spans, span_labels = sentence_tagger.spans
            relations, relation_labels = sentence_tagger.relations
            bio_labels = sentence_tagger.bio_labels


            examples.append(
                InputExample(guid=guid, input_ids=input_ids, text_a=text, words=words, spans=spans, relations=relations, span_labels=span_labels,
                             relation_labels=relation_labels, word_pos_to_token_pos=word_pos_to_token_pos, bio_labels=bio_labels)) # Пользы особой не видно, но ничего в данных не меняте
        
        # print('EXAMPLE', examples[0])
        return examples

    def _read_txt(self, file_path):
        lines = []
        with open(file_path, "r", encoding="utf8") as f:
            data = f.readlines()
            for d in data:
                text, label = d.strip().split("####")
                row = {"text": text, "labels": eval(label)}
                lines.append(row)
        return lines





