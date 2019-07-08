import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from utils import evaluate
from models import RaptorXModel
from reader import Dataset, TEST_MODE

parser = argparse.ArgumentParser(description='Testing Model')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')

# Helper Functions
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main Function
def main():
    create_dir_if_not_exists('predicted_maps/')

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

    # Load dataset
    dataset = Dataset()

    # Test on test split
    accuracy_infos = {}
    for _ in range(dataset.test.size):
        _id, _1d_feature, _2d_feature, contact_map = dataset.next_target(TEST_MODE, cuda, device_id)
        print('Testing on target {}'.format(_id))

        # Apply the trained model
        network_outputs = model(_1d_feature, _2d_feature)
        network_outputs = torch.softmax(network_outputs, -1)

        contact_map = contact_map.cpu().data.numpy().squeeze()
        network_outputs = network_outputs.cpu().data.numpy().squeeze()
        accuracy_info = evaluate(contact_map, network_outputs)
        accuracy_infos[_id] = accuracy_info
        print(accuracy_info)

        np.save('predicted_maps/{}.npy'.format(_id), network_outputs[:,:,1])

    # Save accuracy_infos and calculate final averaged accuracy scores
    with open('predicted_maps/accuracy_infos.json', 'w') as outfile:
        json.dump(accuracy_infos, outfile)
    print('\n')
    for r in [10, 5, 2, 1]:
        for type in ['short', 'medium', 'long']:
            top_l_r = str('L/{}'.format(r))
            score = np.average([info[top_l_r][type] for info in accuracy_infos.values()])
            print('For {} and {}-range contacts: {}'.format(top_l_r, type, score))

if __name__=="__main__":
    main()
