#%%
import numpy as np
import pandas as data
import pandas as pd
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
#%%

#%%
def get_sliding_idx(min_idx, max_idx, window_size, offset):
    idx = []
    for i in range(min_idx, max_idx - offset, offset):
        if i+window_size <= max_idx: idx.append([x for x in range(i, i+window_size)])
    return np.array(idx)

def get_sliding_windows(df, feature_cols = ['x', 'y', 'z'], label_col = 'gt', window_size = 48, offset = 24):
    idx = get_sliding_idx(min_idx=0, max_idx=df.shape[0], window_size=window_size, offset=offset)
    Xs, ys = [], []
    for i in range(0, idx.shape[0]):
        chunk = df.iloc[idx[i]]
        ys.append(chunk[label_col].mode()[0])
        # be very careful whether you are using unstack or stack as it will mess up ordering
        Xs.append(chunk[feature_cols].stack().reset_index(drop=True))
    df = pd.DataFrame(Xs)
    df[df.columns[-1]+1] = ys
    # return pd.DataFrame(Xs), pd.DataFrame(ys)
    # return np.array(Xs), np.array(ys)
    return df

"""
    assumes the data in df are sorted according to time
"""
def train_test_split(df, train_frac=0.8, label_col='gt', time_col='Arrival_Time'):
    train_idx, test_idx = [], []
    for l in df[label_col].unique():
        idx = df[df[label_col] == l].index
        train_idx.append(idx[:int(idx.shape[0] * train_frac)].values)
        test_idx.append(idx[int(idx.shape[0] * train_frac):].values)
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    # need to make sure we did not mess up the ordering of the data in any way
    return df.loc[train_idx].sort_values(time_col), df.loc[test_idx].sort_values(time_col)

def train_val_test_split(df, val_frac=0.2, test_frac=0.2, label_col='gt', time_col='Arrival_Time'):
    train_idx, val_idx, test_idx = [], [], []
    train_frac = 1 - (val_frac + test_frac)
    for l in df[label_col].unique():
        idx = df[df[label_col] == l].index
        train_idx.append(idx[:int(idx.shape[0] * train_frac)].values)
        val_idx.append(idx[int(idx.shape[0] * train_frac):int(idx.shape[0] * (train_frac+val_frac))].values)
        test_idx.append(idx[int(idx.shape[0] * (train_frac+val_frac)):].values)
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    # need to make sure we did not mess up the ordering of the data in any way
    return df.loc[train_idx].sort_values(time_col), df.loc[val_idx].sort_values(time_col), \
        df.loc[test_idx].sort_values(time_col)

def generate_train_test(train, val, test, n_channel=3):
    X_train_sup = train[:,:-1].astype(np.float)
    y_train_sup = train[:,-1].astype(str)
    X_train_usup = val[:,:-1].astype(np.float)
    y_train_usup = val[:,-1].astype(str)
    X_test = test[:,:-1].astype(np.float)
    y_test = test[:,-1].astype(str)

    n_classes = np.unique(y_train_sup).shape[0]
    window_size = int(X_train_sup.shape[1]/n_channel)
    print('we have {} classes and window size of {}'.format(n_classes, window_size))
    return X_train_sup, y_train_sup, X_train_usup, y_train_usup, X_test, y_test, n_classes, window_size
