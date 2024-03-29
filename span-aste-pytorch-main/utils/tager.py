#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/21 14:04
@Desc    ：
==================================================
"""
from enum import IntEnum
from typing import Tuple, List, Text
from pydantic import BaseModel
import numpy as np



class SpanLabel(IntEnum): # Лейблы для классификации спана на аспект/мнение/другое
    INVALID = 0
    ASPECT = 1
    OPINION = 2


class RelationLabel(IntEnum): # Лейблы для классификации пары аспект/мнение
    INVALID = 0
    POS = 1
    NEG = 2
    NEU = 3

class BIOLabel(IntEnum): # Лейблы для BIO классификации слова
    OUTSIDE = 0
    BEG = 1
    INSIDE = 2

class SentimentTriple(BaseModel):
    aspect: List
    opinion: List
    sentiment: Text

    @classmethod
    def from_sentiment_triple(cls, labels: Tuple[List, List, Text]):
        """read from sentiment triple"""

        assert len(labels) == 3
        return cls(
            aspect=labels[0],
            opinion=labels[1],
            sentiment=labels[2]
        )


class SentenceTagger:
    """例句标注"""

    def __init__(self, sentiments: List[SentimentTriple], use_neutral_class, length):
        self.sentiments = sentiments
        self.use_neutral_class = use_neutral_class
        self.length = length

        if use_neutral_class:
            
            self.sentiments_mapping = {
                "POS": RelationLabel.POS.value,
                "NEG": RelationLabel.NEG.value,
                "NEU": RelationLabel.NEU.value
            }
        else:
            self.sentiments_mapping = {
                "POS": RelationLabel.POS.value,
                "NEG": RelationLabel.NEG.value
            }

    @property
    def spans(self):
        spans, span_labels = [], [] # Индексы спанов и их лейблы (аспект/мнение)
        for triplets in self.sentiments:
            spans.append(tuple(triplets.aspect))
            span_labels.append(SpanLabel.ASPECT.value)
            spans.append(tuple(triplets.opinion))
            span_labels.append(SpanLabel.OPINION.value)
        return spans, span_labels

    @property
    def relations(self):

        relations, relation_labels = [], [] # Индексы триплетов и их тональность
        for triplets in self.sentiments:
            relation = []
            relation.extend(triplets.aspect)
            relation.extend(triplets.opinion)
            relation_labels.append(self.sentiments_mapping[triplets.sentiment])

            relations.append(tuple(relation))
        return relations, relation_labels


    @property
    def bio_labels(self):
        bio_labels = np.array([BIOLabel.OUTSIDE.value] * self.length) # Лейблы для BIO классификации
        for triplets in self.sentiments:
            aspect = triplets.aspect
            opinion = triplets.opinion
            
            aspect = list(range(aspect[0], aspect[1]))
            opinion = list(range(opinion[0], opinion[1]))

            if aspect[-1] < self.length and opinion[-1] < self.length:
                bio_labels[aspect[0]] = BIOLabel.BEG.value
                bio_labels[aspect[1:]] = BIOLabel.INSIDE.value

                bio_labels[opinion[0]] = BIOLabel.BEG.value
                bio_labels[opinion[1:]] = BIOLabel.INSIDE.value

        return bio_labels
