# %%
# column name descriptions
# http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

# The 54 columns in the data files are organized as follows:
# 1. timestamp (s)
# 2. activityID (see below for the mapping to the activities)
# 3. heart rate (bpm)
# 4-20. IMU hand
# 21-37. IMU chest
# 38-54. IMU ankle

# The IMU sensory data contains the following columns:
# 1. temperature (Â°C)
# 2-4. 3D-acceleration data (ms-2), scale: Â±16g, resolution: 13-bit
# 5-7. 3D-acceleration data (ms-2), scale: Â±6g, resolution: 13-bit
# 8-10. 3D-gyroscope data (rad/s)
# 11-13. 3D-magnetometer data (Î¼T)
# 14-17. orientation (invalid in this data collection)

# List of activityIDs and corresponding activities:
# 1 lying
# 2 sitting
# 3 standing
# 4 walking
# 5 running
# 6 cycling
# 7 Nordic walking
# 9 watching TV
# 10 computer work
# 11 car driving
# 12 ascending stairs
# 13 descending stairs
# 16 vacuum cleaning
# 17 ironing
# 18 folding laundry
# 19 house cleaning
# 20 playing soccer
# 24 rope jumping
# 0 other (transient activities)

import sys
import matplotlib.pyplot as plt
from data_utils import get_sliding_windows
import numpy as np
import pandas as pd
import os

from data_utils import (get_sliding_windows, train_test_split,
                        train_val_test_split)

path_prefix = 'E:/Desktop/hdcnn-x-master/data/raw/PAMAP2_Dataset/Protocol/'

# high range accelerometer columns, index starts at zero here unlike the description
acc_high_range_idx = [4, 5, 6, 21, 22, 23, 38, 39, 40]

# low range accelerometer columns, index starts at zero here unlike the description
acc_low_range_idx = [7, 8, 9, 24, 25, 26, 41, 42, 43]
label_col = 1

# we're going to use only these labels, index starts at zero here unlike the description
label_filtered = [1, 2, 3, 4, 5, 6, 12, 13, 16, 17, 24]

window_size = 128
n_channel = 9
offset = int(window_size/8)

args = {}
for user in ['subject101',
             'subject102',
             'subject103',
             'subject104',
             'subject105',
             'subject106',
             'subject107',
             'subject108',
             'subject109']:
    file_path = path_prefix + user + '.dat'
    print('Reading file', file_path)

    df = pd.read_csv(file_path, delimiter=' ', header=None)

    # create a mask to only allow the listed labels
    filter_ = np.zeros(df.shape[0], dtype=bool)
    for l in label_filtered:
        filter_ = np.logical_or(filter_, df[label_col] == l)

    # only use the selected labels
    df = df[filter_]
    # filter the columns and drop null values
    df = df[acc_high_range_idx + [label_col] + [0]]
    print('data shape:', df.shape)
    df.dropna(inplace=True)
    print('data shape after dropping null values:', df.shape)

    df, df_t = train_test_split(df,label_col=1, time_col=0)

    print('creating entry for user:', user)
    sw = get_sliding_windows(df, feature_cols=acc_high_range_idx,
                             label_col=label_col, window_size=window_size, offset=offset)
    sw_t = get_sliding_windows(df_t, feature_cols=acc_high_range_idx,
                             label_col=label_col, window_size=window_size, offset=offset)

    # 101
    labels = sw.values[:, -1].reshape(-1, 1).astype(int)
    temp = sw.values[:, :-1].reshape(sw.shape[0], window_size, n_channel)
    labels_t = sw_t.values[:, -1].reshape(-1, 1).astype(int)
    temp_t = sw_t.values[:, :-1].reshape(sw_t.shape[0], window_size, n_channel)

    v_name = 'data_{}_h_train'.format(user)
    v_name_t = 'data_{}_h_test'.format(user)
    print('writing entry', v_name)
    args[v_name] = np.concatenate([temp[:, :, :3].reshape(temp.shape[0], -1), labels], axis=1)
    args[v_name_t] = np.concatenate([temp_t[:, :, :3].reshape(temp_t.shape[0], -1), labels_t], axis=1)

    v_name = 'data_{}_c_train'.format(user)
    v_name_t = 'data_{}_c_test'.format(user)
    print('writing entry', v_name)
    args[v_name] = np.concatenate([temp[:, :,  3:6].reshape(temp.shape[0], -1), labels], axis=1)
    args[v_name_t] = np.concatenate([temp_t[:, :,  3:6].reshape(temp_t.shape[0], -1), labels_t], axis=1)

    v_name = 'data_{}_a_train'.format(user)
    v_name_t = 'data_{}_a_test'.format(user)
    print('writing entry', v_name)
    args[v_name] = np.concatenate([temp[:, :,  6:9].reshape(temp.shape[0], -1), labels], axis=1)
    args[v_name_t] = np.concatenate([temp_t[:, :,  6:9].reshape(temp_t.shape[0], -1), labels_t], axis=1)


print('writing to file')
np.savez_compressed(
    'E:/Desktop/hdcnn-x-master/data/preprocessed/pamap2_all_users.npz', **args)
