import torch
import json
import random
import numpy as np

from os.path import join
from torch import FloatTensor, LongTensor
from utils import AugmentedList, get_features, read_contact_map, read_json, read_target_list

TRAIN_MODE = 0
DEV_MODE = 1
TEST_MODE = 2
BASE_CASP13_DIR = '/net/kihara/home/lai123/shared-scratch/CASP_13'

# Dataset Class
class Dataset:
    def __init__(self):
        self.train = AugmentedList(read_target_list('data/train.list'), True)
        self.dev = AugmentedList(read_target_list('data/dev.list'))
        self.test = AugmentedList(read_target_list('data/test.list'))

    # Support only batch_size = 1
    def next_target(self, mode, cuda, device_id):
        if mode == TRAIN_MODE: target_id = self.train.next_items(1)[0]
        elif mode == DEV_MODE: target_id = self.dev.next_items(1)[0]
        elif mode == TEST_MODE: target_id = self.test.next_items(1)[0]

        _1d_feature, _2d_feature = get_features(target_id)
        contact_map = read_contact_map(target_id)

        # Convert to FloatTensors
        _1d_feature = FloatTensor(np.expand_dims(_1d_feature, 0))
        _2d_feature = FloatTensor(np.expand_dims(_2d_feature, 0))
        contact_map = LongTensor(np.expand_dims(contact_map, 0))

        if cuda:
            _1d_feature = _1d_feature.cuda(device_id)
            _2d_feature = _2d_feature.cuda(device_id)
            contact_map = contact_map.cuda(device_id)

        return target_id, _1d_feature, _2d_feature, contact_map

# CASP13TestDataset Class
class CASP13TestDataset:
    def __init__(self):
        self.features = read_json(join(BASE_CASP13_DIR, 'casp13_features.json'))
        self.l_contacts = read_json(join(BASE_CASP13_DIR, 'casp13_L_contacts.json'))
        self.ml_contacts = read_json(join(BASE_CASP13_DIR, 'casp13_ML_contacts.json'))
        self.targets = AugmentedList(list(self.features.keys()))

        # Build self.target2domains
        self.target2domains = {}
        for target in self.targets.items:
            self.target2domains[target] = []
            for key in self.l_contacts.keys():
                if key.startswith(target):
                    self.target2domains[target].append(key)

        # Build self.domain2ranges
        self.domain2ranges = {}
        f = open(join(BASE_CASP13_DIR, 'domains.txt'), 'r')
        for line in f:
            domain, ranges = line.strip().split(':')
            ranges = ranges.strip().split(',')
            self.domain2ranges[domain] = []
            for range_str in ranges:
                start_nb, end_nb = range_str.split('-')
                self.domain2ranges[domain].append((int(start_nb), int(end_nb)))

    # Support only batch_size = 1
    def next_target(self, cuda=False, device_id=None):
        target_id = self.targets.next_items(1)[0]
        feature = self.features[target_id]
        _1d_feature = FloatTensor(np.expand_dims(np.transpose(feature['f_1d'], [1, 0]), 0))
        _2d_feature = FloatTensor(np.expand_dims(np.transpose(feature['f_2d'], [1, 2, 0]), 0))
        domains = self.target2domains[target_id]

        ml_contacts, l_contacts = {}, {}
        for domain in domains:
            ml_contacts[domain] = self.ml_contacts[domain]
            l_contacts[domain] = self.l_contacts[domain]

        return target_id, domains, _1d_feature, _2d_feature, ml_contacts, l_contacts
