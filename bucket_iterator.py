# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='char_indices', shuffle=True, sort=True, max_len=70):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches


    def pad_data(self, batch_data):
        batch_char_indices = []
        batch_other_feature = []
        batch_tags= []
        # print(batch_data)
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            # print(item)
            char_indices,other_feature, tags = item['char_indices'], item['other_feature'],item["tags"]
            text_padding = [0] * (max_len - len(char_indices))

            batch_char_indices.append(char_indices + text_padding)
            batch_other_feature.append(other_feature)
            batch_tags.append(tags)

        return { \
            'batch_char_indices': torch.tensor(batch_char_indices), \
            'batch_other_feature': torch.tensor(batch_other_feature), \
            'batch_tags': torch.tensor(batch_tags), \
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
