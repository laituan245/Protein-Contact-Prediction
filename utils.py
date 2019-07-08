import torch
import dgl
import json
import numpy as np
import networkx as nx
import random

from os.path import join
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

# Constants
FEATURES_BASE_DIR = '/net/kihara/home/jain163/Desktop/Projects/folding/Z_Project/feature/feature_out'
CONTACT_MAPS_BASE_DIR = '/net/kihara/home/jain163/Desktop/Projects/folding/Z_Project/contact_map_and_final_fastaseq/contact_map_matrix/'

def evaluate(true_map, predicted_map):
    # Squeeze the maps
    true_map = true_map.squeeze().astype(float)
    predicted_map = predicted_map.squeeze()[:, :, 1].astype(float)
    assert(len(true_map.shape) == 2)
    assert(len(predicted_map.shape) == 2)

    accuracy_info = {}
    protein_len = true_map.shape[0]
    for r in [10, 5, 2, 1]:
        L_r = int(round(protein_len / r))
        accuracy_info[str('L/{}'.format(r))] = {}
        for contact_type in ['short', 'medium', 'long']:
            if contact_type == 'short':
                min_separation, max_separation = 6, 11
            if contact_type == 'medium':
                min_separation, max_separation = 12, 23
            if contact_type == 'long':
                min_separation, max_separation = 24, None

            # Get top pairs
            nb_founds, predicted_top_pairs = get_top_pairs(predicted_map, L_r, min_separation, max_separation)

            # Count number of true positives
            true_positives = 0
            for i in range(nb_founds):
                idx_x, idx_y = predicted_top_pairs[0][i], predicted_top_pairs[1][i]
                if true_map[idx_x, idx_y] == 1.0:
                    true_positives += 1

            # Update accuracy_info
            accuracy_info[str('L/{}'.format(r))][contact_type] = true_positives / L_r
    return accuracy_info

def read_json(fn):
    with open(fn) as f:
        return json.load(f)

def read_target_list(list_fn):
    targets = []
    f = open(list_fn, 'r')
    for line in f:
        targets.append(line.strip())
    f.close()
    return targets

def read_contact_map(id):
    contact_map_file = join(CONTACT_MAPS_BASE_DIR, '{}_contact_map.npy'.format(id))
    contact_map = np.load(contact_map_file)
    return contact_map

def get_features(target):
    # Get 1D feature
    _1d_feature_fp = join(FEATURES_BASE_DIR, '{}.1d_feature.npy'.format(target))
    _1d_feature = np.load(_1d_feature_fp)
    _1d_feature = np.transpose(_1d_feature, [1, 0])

    # Get 2D feature
    _2d_feature_fp = join(FEATURES_BASE_DIR, '{}.2d_feature.npy'.format(target))
    _2d_feature = np.load(_2d_feature_fp)
    _2d_feature = np.transpose(_2d_feature, [1, 2, 0])

    return _1d_feature, _2d_feature

def get_top_pairs(mat, num_contacts, min_separation, max_separation = None):
    """Get the top-scoring contacts"""
    idx_delta = np.arange(mat.shape[1])[np.newaxis, :] - np.arange(mat.shape[0])[:, np.newaxis]

    if max_separation:
        mask = (idx_delta < min_separation) | (idx_delta > max_separation)
    else:
        mask = idx_delta < min_separation

    mat_masked = np.copy(mat)
    mat_masked[mask] = float("-inf")

    top = mat_masked.argsort(axis=None)[::-1][:(num_contacts)]
    top = (top % mat.shape[0]).astype(np.uint16), np.floor(top / mat.shape[0]).astype(np.uint16)

    # Post-filtering
    filtered_indices_x, filtered_indices_y, num_contacts_found = [], [], 0
    indices_x, indices_y = top
    for i in range(num_contacts):
        index_x, index_y = indices_x[i], indices_y[i]
        dist = abs(index_x.astype(float) - index_y.astype(float))
        if dist < min_separation or (max_separation != None and dist > max_separation): continue
        num_contacts_found += 1
        filtered_indices_x.append(index_x)
        filtered_indices_y.append(index_y)

    return num_contacts_found, (filtered_indices_x, filtered_indices_y)

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch
        if shuffle_between_epoch:
            random.shuffle(self.items)

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        return len(self.items)
