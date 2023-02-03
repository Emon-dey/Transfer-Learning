# %%
import sys

import glob
import os
import pandas as pd
import numpy as np
from data_utils import get_sliding_windows
import matplotlib.pyplot as plt


path_prefix = 'data/raw/OpportunityUCIDataset/dataset/'
label_col = 243
# BACK  - 37, 38, 39
# RUA   - 50, 51, 52
# RLA   - 63, 64, 65
# LUA   - 76, 77, 78
# LLA   - 89, 90, 91
feature_cols = [37, 38, 39, 50, 51, 52, 63, 64, 65, 76, 77, 78, 89, 90, 91]

window_size = 128
n_channel = 15  # 5 sensors * 3 accelerometer channels
offset = int(window_size/8)

# %%

args = {}
for i in range(1, 5):
    user = 'S' + str(i)
    regex = 'S{}-*.dat'.format(i)
    l = []
    for item in sorted(glob.glob(path_prefix + regex)):
        print('Reading file:', item)
        l.append(pd.read_csv(item, delimiter=' ', header=None))
    df = pd.concat(l)

    # only select data with locomotive labels
    df = df[df[label_col] > 0]

    # lower the range of the values
    df2 = df[feature_cols] / 1000.
    df = pd.concat([df2, df[label_col]], axis=1)

    # remove rows with null value
    print('data shape:', df.shape)
    df.dropna(inplace=True)
    print('data shape after dropping null values:', df.shape)

    print('creating entry for user:', user)
    sw = get_sliding_windows(df, feature_cols=feature_cols,
                             label_col=label_col, window_size=window_size, offset=offset)

    # extract the label column and make it int
    labels = sw.values[:, -1].reshape(-1, 1).astype(int)

    # reshape the sliding window for more predictable data extraction per sensor
    temp = sw.values[:, :-1].reshape(sw.shape[0], window_size, n_channel)

    for j, pos in enumerate(['BACK', 'RUA', 'RLA', 'LUA', 'LLA']):
        v_name = 'data_{}_{}'.format(user, pos)
        print('writing entry', v_name)
        args[v_name] = np.concatenate(
            [temp[:, :, j*3:(j+1)*3].reshape(temp.shape[0], -1), labels], axis=1)

print('Writing final file')
np.savez_compressed(
    'phd/strangan/data/preprocessed/opportunity_all_users.npz', **args)
