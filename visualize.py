# Pick 5 random targets from the test set (data/test.list).
# And create a plot for comparing between the true contact maps and predicted
# contact maps of the randomly selected targets.

import os
import random
import numpy as np
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from utils import read_target_list

CONTACT_MAPS_BASE_DIR = '/net/kihara/home/jain163/Desktop/Projects/folding/Z_Project/contact_map_and_final_fastaseq/contact_map_matrix/'
NB_TARGETS = 5

# Main code
test_targets = read_target_list('data/test.list')
random.shuffle(test_targets)
selected_targets = test_targets[:NB_TARGETS]

# create a figure
fig, ax = plt.subplots(NB_TARGETS, 2, figsize=(12, 12))
ax[0, 0].set_title('True Map')
ax[0, 1].set_title('Predicted Map')

# show in figure 10 random 64x64 slices
i = 0
for target in selected_targets:
    contact_map = np.load(join(CONTACT_MAPS_BASE_DIR, '{}_contact_map.npy'.format(target)))
    predicted_map = np.load('predicted_maps/{}.npy'.format(target))
    im1 = ax[i, 0].imshow(contact_map, interpolation='none')
    im2 = ax[i, 1].imshow(predicted_map, interpolation='none')
    i += 1

plt.savefig('plot.png')
