import torch


def gold_labels(span_indices, spans, span_labels):
    """
    Organizing gold labels and indices
    :param span_indices: Индексы всех спанов [[(0, 0), (1, 1), (0, 1)], [(0, 0), (1, 1), (0, 1)]] или пар спанов (при классификации пар аспект/мнение) [[(0, 0, 1, 1)], [(0, 0, 1, 1)]] (Список списков кортежей)
    :param spans: Индексы таргет спанов ([(0, 1)], [(0, 0)]) или пар аспект/мнение ([(0, 0, 1, 1)], [(0, 0, 1, 1)]) (Кортеж списков кортежей)
    :param span_labels: Лейблы таргет спанов ([1, 2, 1, 2], [2, 1, 2]) или пар аспект/мнение ([1, 3], [2, 2])
    :return:
        gold_labels:
    """
    gold_labels = [] # Лейблы спанов или пар аспект/мнение: [[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]]


    for batch_idx, indices in enumerate(span_indices): # Итерируемся по примерам в батче
        gold_lab = [0] * len(indices)
        for gold_span_idx, span in enumerate(spans[batch_idx]): # Итерируемся по таргетам
          try:
              index = indices.index(span) # Проверяем, что таргет есть во всех возможных спанах или парах аспект/мнение (Его может не быть, если в таргете слово после 512 токенов)
              
              gold_lab[index] = span_labels[batch_idx][gold_span_idx]
          except ValueError: # Если таргета нет во всех спанах или парах аспект/мнение
              pass

        gold_labels.append(gold_lab)

    return gold_labels


def collate_fn(data):
    """批处理，填充同一batch中句子最大的长度"""
    # Небольшая заглушка для формирования батча
    return zip(*data)
    
    # return input_ids, attention_mask, words, spans, relations, span_labels, relation_labels, word_pos_to_token_pos, bio_labels
