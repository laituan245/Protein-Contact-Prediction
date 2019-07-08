import itertools
import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from models import RaptorXModel
from reader import CASP13TestDataset 

parser = argparse.ArgumentParser(description='Testing Model')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')

def evaluate(ranges, ml_contacts, predicted_map):
    all_pairs = []
    nb_ranges = len(ranges)
    for i in range(nb_ranges):
        for j in range(i, nb_ranges):
            list_i = list(range(ranges[i][0]-1, ranges[i][1]))
            list_j = list(range(ranges[j][0]-1, ranges[j][1]))
            _pairs = itertools.product(list_i, list_j)
            all_pairs = all_pairs + list(_pairs)
    all_pairs = [pair for pair in all_pairs if pair[0] <= pair[1]]

    L = 0
    for i in range(nb_ranges):
        L += (ranges[i][1] - ranges[i][0] + 1)
    all_pairs.sort(key=lambda x: predicted_map[x[0], x[1]], reverse=True)

    accuracy_info = {}
    for r in [5, 2, 1]:
        accuracy_info[str('L/{}'.format(r))] = {}
        for contact_type in ['ml', 'l']:
            L_r = int(round(L / r))
            if contact_type == 'ml': min_separation = 12
            if contact_type == 'l': min_separation = 24

            # Get top pairs satisfying the conditions
            predicted_top_pairs = []
            for i in range(len(all_pairs)):
                if all_pairs[i][1] - all_pairs[i][0] >= min_separation:
                    predicted_top_pairs.append((all_pairs[i][0], all_pairs[i][1]))
                    if len(predicted_top_pairs) == L_r: break

            # Count number of true positives
            true_positives = 0
            for pair in predicted_top_pairs:
                idx_x, idx_y = pair[0]+1, pair[1]+1
                if str(idx_x) in ml_contacts and idx_y in ml_contacts[str(idx_x)]:
                    true_positives += 1

            # Update accuracy_info
            accuracy_info[str('L/{}'.format(r))][contact_type] = true_positives / L_r
    return accuracy_info

def main():
    # Arguments Parsing
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true' and torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    device_id = args.device_id

    # Load model
    model = RaptorXModel(feature_1d_dim=24, feature_2d_dim=3)
    if cuda:
        model.cuda(device_id)
    model = torch.load('model/raptorx_model')
    model.eval()
    print('Loaded model')

    # Load dataset
    dataset = CASP13TestDataset()
    print('Loaded CASP13 test dataset')

    # Start testing
    accuracy_infos = {}
    domain2ranges = dataset.domain2ranges
    for _ in range(dataset.targets.size):
        target_id, domains, _1d_feature, _2d_feature, ml_contacts, l_contacts = dataset.next_target(cuda, device_id)

        network_outputs = model(_1d_feature, _2d_feature)
        network_outputs = torch.softmax(network_outputs, -1)
        network_outputs = network_outputs.cpu().data.numpy().squeeze()

        # Evaluation
        for domain in domains:
            print('Domain: {}'.format(domain))
            ranges = domain2ranges[domain]
            _ml_contacts = ml_contacts[domain]
            predicted_map = network_outputs[:,:,1]
            accuracy_info = evaluate(ranges, _ml_contacts, predicted_map)
            print(accuracy_info)
            accuracy_infos[domain] = accuracy_info
    for r in [5, 2, 1]:
        for type in ['ml', 'l']:
            top_l_r = str('L/{}'.format(r))
            score = np.average([info[top_l_r][type] for info in accuracy_infos.values()])
            print('For {} and {}-range contacts: {}'.format(top_l_r, type, score))

if __name__=="__main__":
    main()
