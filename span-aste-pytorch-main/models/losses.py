# Не используется, легче считать лосс напрямую через CrossEntropyLoss, иначе несколько часов эпоха длится




# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# """
# =================================================
# @Project ：span-aste
# @IDE     ：PyCharm
# @Author  ：Mr. Wireless
# @Date    ：2022/1/19 11:19 
# @Desc    ：
# ==================================================
# """
# import torch
# import time


# def log_likelihood(criterion, probability, gold_labels):
#     """
#     The training objective is defined as the sum of the negative log-likelihood from both the mention module and triplet module.
#     where m∗i,j is the gold mention type of the span si,j ,and r∗is the gold sentiment relation of the target and opinion span
#     pair (St_a,b, So_c,d). S indicates the enumerated span pool; Stand So are the pruned target and opinion span candidates.
#     :param probability: the probability from span or candidates
#     :type Tensor
#     :param indices: the indices for predicted span or candidates
#     :type List[List[Tuple(i,j)]] or List[List[Tuple(a,b,c,d)]]
#     :param span:
#     :param labels:
#     :type List[List[0/1)]]
#     :return: negative log-likelihood
#     """
#     # print(probability.shape)
#     # print(probability)
#     # assert indices == gold_indices
#     # print(indices)
#     # print(gold_indices)
#     # print(gold_labels)
#     # Statistically predict the indices of the correct mention or candidates

#     # import pickle
#     # with open('probability.pickle', 'wb') as f:
#     #   pickle.dump(probability, f)
#     # with open('indices.pickle', 'wb') as f:
#     #   pickle.dump(indices, f)
#     # with open('gold_labels.pickle', 'wb') as f:
#     #   pickle.dump(gold_labels, f)
#     # 1/0


#     shape = probability.shape
#     device = probability.device
#     probability = torch.cat([probability[batch_idx] for batch_idx in range(shape[0])])
#     # probability = probability.reshape(shape[0]*shape[1], shape[2])
#     gold_labels = torch.tensor(gold_labels, device=device).reshape(-1)
#     # gold_indice_labels = []
#     # for batch_idx, label in enumerate(gold_indices):
#     #     for i, l in enumerate(label):
#     #         # if l in indices[batch_idx]:
#     #         #     idx = indices[batch_idx].index(l)
#     #         #     gold_indice_labels.append((batch_idx, idx, gold_labels[batch_idx][i]))
#     #         gold_indice_labels.append((batch_idx, i, gold_labels[batch_idx][i]))

#     # # sum of the negative log-likelihood from both the mention module and triplet module
#     # loss = [-torch.log(probability[c[0], c[1], c[2]]) for c in gold_indice_labels]
#     # print(probability.shape, gold_labels.shape)


#     loss = criterion(torch.log(probability), gold_labels)
#     # loss = -sum(torch.log(probability[torch.arange(gold_labels.shape[0]), gold_labels]))


#     # print(time.time())
#     # loss = torch.stack(loss).sum()
#     return loss
