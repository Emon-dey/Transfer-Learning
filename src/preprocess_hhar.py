# %%
import os
import sys

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import (get_sliding_windows, train_test_split,
                        train_val_test_split)

path_prefix = '/notebooks/hdcnn-x-master/data/raw/Activity recognition exp/'

phones_acc = pd.read_csv(path_prefix + 'Phones_accelerometer.csv', index_col=0)
phones_acc.drop('Creation_Time', inplace=True, axis=1)
phones_acc.sort_values('Arrival_Time', inplace=True)
phones_acc.fillna('nothing', inplace=True)
phones_acc['Arrival_Time'] = pd.to_numeric(phones_acc['Arrival_Time']).apply(lambda x: datetime.fromtimestamp(x/1e3))

subjects = list('abcdefghi')
devices = ['nexus4_1, s3_2, s3mini_1, samsungold_1']

window_size = 128
n_channel = 9
offset = int(window_size/8)

args = {}
idx_act = (phones_acc['gt'] != 'nothing')
for user in list('abcdefghi'):
    idx_user = (phones_acc['User'] == user)
    for device in ['nexus4_1', 's3_2', 's3mini_1', 'samsungold_1']:
        v_name = 'data_' + user + '_' + device.replace('_1', '').replace('_2', '') + '_train'
        vt_name = 'data_' + user + '_' + device.replace('_1', '').replace('_2', '') + '_test'
        print('Creating variable {}'.format(v_name))
        d = phones_acc[(phones_acc['Device'] == device) & idx_user & idx_act]
        d, d_t = train_test_split(d)
        

        sw = get_sliding_windows(d, window_size=window_size, offset=offset).sample(frac=1).reset_index(drop=True)
        sw_t = get_sliding_windows(d_t, window_size=window_size, offset=offset).sample(frac=1).reset_index(drop=True)
        args[v_name] = sw
        args[vt_name] = sw_t

np.savez_compressed('/notebooks/hdcnn-x-master/data/preprocessed/hhar_all_users.npz', **args)