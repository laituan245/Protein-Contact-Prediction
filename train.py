import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from models import RaptorXModel
from reader import Dataset, TRAIN_MODE, DEV_MODE, TEST_MODE

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--iterations', type=int, default=10000, help='Number training iterations')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizers')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')

# Helper Functions
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main Function
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

    save_path = 'model/raptorx_model'
    batch_size = 1
    iterations = args.iterations
    learning_rate = args.learning_rate
    log_interval = args.log_interval

    # Load dataset
    dataset = Dataset()

    # Load model
    model = RaptorXModel(feature_1d_dim=24, feature_2d_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if cuda:
        model.cuda(device_id)
    criterion = nn.CrossEntropyLoss()

    # Start Training
    for itx in range(iterations):
        model.train()
        model.zero_grad()

        _, _1d_feature, _2d_feature, contact_map = dataset.next_target(TRAIN_MODE, cuda, device_id)

        network_outputs = model(_1d_feature, _2d_feature)
        loss = criterion(network_outputs.view(-1, 2), contact_map.view(-1))
        loss.backward()
        optimizer.step()
        print('loss = {}'.format(loss))

        if itx > 0 and itx % args.log_interval == 0:
            create_dir_if_not_exists('model')
            torch.save(model, save_path)
            print('Saved the model')
if __name__=="__main__":
    main()
