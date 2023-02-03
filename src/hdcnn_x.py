# %%
import argparse
import csv
import os
import sys
from pprint import pprint

#import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from IPython import embed
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from torch.optim import SGD, Adam, Adamax
from torch.utils.data import DataLoader, Dataset
#from torchsummary import summary

eps = 1e-8

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ActivityDataset(data.Dataset):
    def __init__(self, datamat, window_size, n_channel, scaling=False, lencoder=None):
        self.n_channel = n_channel
        self.window_size = window_size
        self.scaler = StandardScaler()

        self.X = datamat[:, :-1].astype(np.float)
        if scaling:
            self.X = self.scaler.fit_transform(self.X)
        self.X = self.X.reshape(self.X.shape[0], -1, 1, self.n_channel)
        # output shape after last line: (samples, window, 1, channel)
        self.X = self.X.transpose(0, 3, 1, 2)  # output (samples, channel, window, 1)

        if lencoder is not None:
            self.lencoder = lencoder
            self.y = self.lencoder.transform(datamat[:, -1].astype(str))
        else:
            self.lencoder = LabelEncoder()
            self.y = self.lencoder.fit_transform(datamat[:, -1].astype(str))

    def __getitem__(self, index):
        sample, target = self.X[index], self.y[index]
        # if needed reshape the sample
        return sample, target

    def __len__(self):
        return self.X.shape[0]


def kl_div(y_pred, y_true, clip=True, mask=[0, 1, 1, 0, 1]):
    r"""
    the Kullback–Leibler divergence of Q from P is defined to be
          D_KL (P||Q) = Sum (P(x) * log (P(x) / Q(x)))
    we assume P is the target, Q is the prediction
    """

    if clip:
        y_pred = torch.clamp(y_pred, eps, 1)
        y_true = torch.clamp(y_true, eps, 1)

    # original formula, taken from keras
    return torch.sum(y_true * torch.log(y_true / y_pred))


def js_div(y_pred, y_true):
    """proper implementation as per wikipedia formula"""

    y_pred = torch.clamp(y_pred, eps, 1)
    y_true = torch.clamp(y_true, eps, 1)
    m = (y_true + y_pred) / 2

    return torch.sum(y_pred * torch.log(y_pred / m) + y_true * torch.log(y_true / m)) / 2


def js_div_keras(y_pred, y_true):
    """copied from my old keras implementation"""
    y_pred = torch.clamp(y_pred, eps, 1)
    y_true = torch.clamp(y_true, eps, 1)

    return torch.sum(y_true * torch.log(y_true / y_pred) + y_pred * torch.log(y_pred / y_true))


def cross_entropy(y_pred, y_true):
    """
    cross entropy of distribution q (y_pred) relative to p (y_true)
        H(p, q) = H(p) + D_KL(p||q)
    """
    # the H(p) term is negated by the first halp of D_KL
    return - torch.sum(y_true * torch.log(y_pred))


def normalize(grad):
    my_min = torch.min(grad)
    my_max = torch.max(grad)
    my_range = my_max - my_min
    return (grad - my_min) / my_range


class Net_FC(nn.Module):
    def __init__(self, n_channel=3, n_classes=6):
        super(Net_FC, self).__init__()
        self.n_channel = n_channel
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=(9, 1))
        self.conv1e = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1))
        self.conv2e = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1))
        self.conv3e = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*9*1, out_features=64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(in_features=64, out_features=self.n_classes)

    def compute_rank_1(self, grad):
        self.conv1_grads = grad

    def compute_rank_2(self, grad):
        self.conv2_grads = grad

    def compute_rank_3(self, grad):
        self.conv3_grads = grad

    def forward(self, x):                   # input: (batch, channel, window, 1)
        out_conv1 = self.conv1(x)
        out_conv1.register_hook(self.compute_rank_1)
        x = self.conv1e(out_conv1)
        # out_conv1 = F.softmax(out_conv1.view(-1, 64*120*1), dim=1)

        out_conv2 = self.conv2(x)
        out_conv2.register_hook(self.compute_rank_2)
        x = self.conv2e(out_conv2)
        # out_conv2 = F.softmax(out_conv2.view(-1, 128*52*1), dim=1)

        out_conv3 = self.conv3(x)
        out_conv3.register_hook(self.compute_rank_3)
        x = self.conv3e(out_conv3)
        # out_conv3 = F.softmax(out_conv3.view(-1, 256*18*1), dim=1)

        x = x.view(-1, 256*9*1)

        out_fc1 = self.fc1(x)
        out_fc2 = self.fc2(out_fc1)
        out_softmax = F.log_softmax(out_fc2, dim=1)

        return out_conv1, out_conv2, out_conv3, out_softmax  # out_fc1, out_fc2,


class HDCNN(object):
    def __init__(self, device, data_source, data_target, p):
        self.n_classes = p.n_classes
        self.n_channel = p.n_channels
        self.window_size = p.window_size
        self.device = device
        self.log_interval = p.log_interval

        self.w_conv1 = p.w_conv1
        self.w_conv2 = p.w_conv2
        self.w_conv3 = p.w_conv3
        self.w_fc1 = p.w_fc1
        self.w_fc2 = p.w_fc2

        # -------------------------------------------------------
        # define networks
        # -------------------------------------------------------
        self.net_FCs = Net_FC(self.n_channel, self.n_classes).to(device)
        self.net_FCt = Net_FC(self.n_channel, self.n_classes).to(device)

        # -------------------------------------------------------
        # init weights
        # -------------------------------------------------------
        self.net_FCs.apply(weight_init)
        self.net_FCt.apply(weight_init)

        # -------------------------------------------------------
        # define optimizer and
        # -------------------------------------------------------
        self.optim_FCs = Adam(self.net_FCs.parameters(), lr=p.lr_FCs, betas=(p.lr_FCs_b1, p.lr_FCs_b2), amsgrad=False)
        self.optim_FCt = Adam(self.net_FCt.parameters(), lr=p.lr_FCt, betas=(p.lr_FCt_b1, p.lr_FCt_b2), amsgrad=False)
        # self.optim_FCs = SGD(self.net_FCs.parameters(), lr=p.lr_FCs, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # self.optim_FCt = SGD(self.net_FCt.parameters(), lr=p.lr_FCt, momentum=0.9, weight_decay=1e-4, nesterov=True)

        self.model_prefix = 'model'

        # -------------------------------------------------------
        # define data loaders
        # -------------------------------------------------------
        # loaders used for training
        self.source_loader = DataLoader(ActivityDataset(data_source, p.window_size, p.n_channels, p.scaling),
                                        batch_size=p.batch_size, shuffle=True, drop_last=True)
        self.target_loader = DataLoader(ActivityDataset(data_target, p.window_size, p.n_channels, p.scaling),
                                        batch_size=p.batch_size, shuffle=True, drop_last=True)

        # loaders used for evaluation, has a larger batch size
        self.source_loader_eval = DataLoader(ActivityDataset(
            data_source, p.window_size, p.n_channels, p.scaling), batch_size=p.batch_size_eval, shuffle=True, drop_last=False)
        self.target_loader_eval = DataLoader(ActivityDataset(
            data_target, p.window_size, p.n_channels, p.scaling), batch_size=p.batch_size_eval, shuffle=True, drop_last=False)

    def train_source(self, p):
        for epoch in range(1, p.source_epochs+1):
            self.net_FCs.train()
            for batch_idx, (data, target) in enumerate(self.source_loader):
                data, target = data.to(self.device).float(), target.to(self.device)
                self.optim_FCs.zero_grad()
                out_conv1, out_conv2, out_conv3, output = self.net_FCs(data)  # out_fc1, out_fc2,
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optim_FCs.step()
                if batch_idx % p.log_interval == 0:
                    logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.source_loader.dataset),
                        100. * batch_idx / len(self.source_loader), loss.item()))
            # test after each epoch
            logging.debug('--------------------------------------------------------------------------------')
            self.test(self.net_FCs, self.source_loader_eval, 'SOURCE            :')
            self.test(self.net_FCs, self.target_loader_eval, 'TARGET w/o tranfer:')
            logging.debug('--------------------------------------------------------------------------------')

    def train_target(self, p):
        # -----------------------------------------------------
        #  Copy final layer weights & bias from source network
        #  Then freeze the weight and bias
        # -----------------------------------------------------
        # self.net_FCt.fc1[0].weight.data.copy_(self.net_FCs.fc1[0].weight.data)
        # self.net_FCt.fc1[0].bias.data.copy_(self.net_FCs.fc1[0].bias.data)
        # self.net_FCt.fc1[0].weight.requires_grad = False
        # self.net_FCt.fc1[0].bias.requires_grad = False

        # self.net_FCt.fc2.weight.data.copy_(self.net_FCs.fc2.weight.data)
        # self.net_FCt.fc2.bias.data.copy_(self.net_FCs.fc2.bias.data)
        # self.net_FCt.fc2.weight.requires_grad = False
        # self.net_FCt.fc2.bias.requires_grad = False
        # -----------------------------------------------------

        for epoch in range(1, p.target_epochs+1):
            self.net_FCt.train()
            self.net_FCs.eval()
            for batch_idx, (data, target) in enumerate(self.target_loader):
                data, target = data.to(self.device).float(), target.to(self.device)

                # ---------------------------------------------
                #  Get the source activations with target data
                # ---------------------------------------------

                self.net_FCs.zero_grad()  # make sure we don't have gradient accumulation
                # outS_conv1 shape: (32, 64, 60, 1)
                outS_conv1, outS_conv2, outS_conv3, outS = self.net_FCs(data.detach())  # outS_fc1, outS_fc2
                lossS = F.nll_loss(outS, target)
                lossS.backward()

                # ----------------------------
                #  Train the target classifier
                # ----------------------------
                self.optim_FCt.zero_grad()
                outT_conv1, outT_conv2, outT_conv3, outT = self.net_FCt(data)  # outT_fc1, outT_fc2,
                loss = F.nll_loss(outT, target)
                loss.backward(retain_graph=True)

                if p.mode == 'train':
                    loss_conv1 = loss_conv2 = loss_conv3 = torch.Tensor([-1])
                elif p.mode == 'hdcnn':
                    # loss_conv1 = kl_div(outT_conv1, outS_conv1.data)
                    # loss_conv2 = kl_div(outT_conv2, outS_conv2.data)
                    # loss_conv3 = kl_div(outT_conv3, outS_conv3.data)

                    loss_conv1 = kl_div(F.softmax(outT_conv1.view(-1, 64*120*1), dim=1),
                                        F.softmax(outS_conv1.data.view(-1, 64*120*1), dim=1))
                    loss_conv2 = kl_div(F.softmax(outT_conv2.view(-1, 128*52*1), dim=1),
                                        F.softmax(outS_conv2.data.view(-1, 128*52*1), dim=1))
                    loss_conv3 = kl_div(F.softmax(outT_conv3.view(-1, 256*18*1), dim=1),
                                        F.softmax(outS_conv3.data.view(-1, 256*18*1), dim=1))
                elif p.mode == 'prune':

                    loss_conv1 = kl_div(F.softmax((outT_conv1 * normalize(self.net_FCt.conv1_grads)).view(-1, 64*120*1), dim=1),
                                        F.softmax((outS_conv1.data * normalize(self.net_FCs.conv1_grads)).data.view(-1, 64*120*1), dim=1))
                    loss_conv2 = kl_div(F.softmax((outT_conv2 * normalize(self.net_FCt.conv2_grads)).view(-1, 128*52*1), dim=1),
                                        F.softmax((outS_conv2.data * normalize(self.net_FCs.conv2_grads)).data.view(-1, 128*52*1), dim=1))
                    loss_conv3 = kl_div(F.softmax((outT_conv3 * normalize(self.net_FCt.conv3_grads)).view(-1, 256*18*1), dim=1),
                                        F.softmax((outS_conv3.data * normalize(self.net_FCs.conv3_grads)).data.view(-1, 256*18*1), dim=1))

                    # loss_conv1 = kl_div(outT_conv1 * normalize(self.net_FCt.conv1_grads), outS_conv1.data * normalize(self.net_FCs.conv1_grads))
                    # loss_conv2 = kl_div(outT_conv2 * normalize(self.net_FCt.conv2_grads), outS_conv2.data * normalize(self.net_FCs.conv2_grads))
                    # loss_conv3 = kl_div(outT_conv3 * normalize(self.net_FCt.conv3_grads), outS_conv3.data * normalize(self.net_FCs.conv3_grads))
                    # loss_fc1 = kl_div(outT_fc1, outS_fc1)
                    # loss_fc2 = kl_div(outT_fc2, outS_fc2)
                    loss_conv = (loss_conv1 * self.w_conv1 + loss_conv2 *
                                 self.w_conv2 + loss_conv3 * self.w_conv3) / p.batch_size
                    loss_conv.backward()

                self.optim_FCt.step()

                if batch_idx % p.log_interval == 0:
                    logging.debug('[Epoch {}/{}] [Iter {}/{}]\t[Losses conv1: {:.4f} | conv2: {:.4f} | conv3: {:.4f} | final: {:.8f}]'.format(  # FC1: {:.4f} | FC2: {:.4f} |
                        epoch, p.target_epochs,
                        batch_idx * len(data), len(self.target_loader.dataset),
                        loss_conv1.item(), loss_conv2.item(), loss_conv3.item(), loss.item()
                        # loss_fc1.item(), loss_fc2.item(),
                    ))
            logging.debug('--------------------------------------------------------------------------------')
            self.test(self.net_FCt, self.target_loader_eval, 'TARGET with tranfer:')
            logging.debug('--------------------------------------------------------------------------------')

    def test(self, model, test_loader, title=''):
        model.eval()
        test_loss = 0
        correct = 0
        y_true, y_pred = [], []
        # with torch.no_grad():
        for data, target in test_loader:
            data = data.to(self.device).float()
            target = target.to(self.device)
            y_true.append(target)

            output = model(data)[-1]

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='mean').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.append(pred.view_as(target))

            correct += pred.eq(target.view_as(pred)).sum().item()
            model.zero_grad()
        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        y_true = torch.cat(y_true, 0).cpu().numpy()
        y_pred = torch.cat(y_pred, 0).cpu().numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='micro')

        logging.debug('{} [loss: {:.6f}] [acc: {}/{} ({:.2f}%)] [f1: {:.4f}]'.format(
            title, test_loss, correct, len(test_loader.dataset), 100. * acc, f1))

        return test_loss, acc, precision, recall, f1  # , support


parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)


# overall dataset params
# --------------------------------------------------------------------------------------------------
parser.add_argument('-ch',  '--n_channels', type=int, default=3, help='Number of channels in the activity data stream')
parser.add_argument('-cls', '--n_classes', type=int, default=6, help='Number of classes in the activity data stream')
parser.add_argument('-bs',  '--batch_size', type=int, default=32)
parser.add_argument('-bse', '--batch_size_eval', type=int, default=1024)
parser.add_argument('-ws',  '--window_size', type=int, default=128, help='Shape of the sliding window')
parser.add_argument('-wo',  '--window_offset', type=int, default=8, help='Stride of the sliding window')
parser.add_argument('-es', '--source_epochs', type=int, default=5, help='Number of source epochs')
parser.add_argument('-et', '--target_epochs', type=int, default=5, help='Number of target epochs')
parser.add_argument('-i', '--n_iter', type=int, default=20000, help='Number of iteration cycles')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval_interval', type=int, default=500)
parser.add_argument('--gpu', default='0')

parser.add_argument('--mode', default='prune', help='train, hdcnn, prune')

# learning rates
# --------------------------------------------------------------------------------------------------
# Source network
parser.add_argument('--lr_FCs',    type=float, default=0.002,  help='Source learning rate')
parser.add_argument('--lr_FCs_b1', type=float, default=0.9,      help='Source learning rate beta1')
parser.add_argument('--lr_FCs_b2', type=float, default=0.999,    help='Source learning rate beta2')

# Target Network
parser.add_argument('--lr_FCt',    type=float, default=0.0002,  help='Target learning rate')
parser.add_argument('--lr_FCt_b1', type=float, default=0.9,      help='Target learning rate beta1')
parser.add_argument('--lr_FCt_b2', type=float, default=0.999,    help='Target learning rate beta2')

# Loss weights
# --------------------------------------------------------------------------------------------------
parser.add_argument('--w_conv1',    type=float, default=0.5,        help='')
parser.add_argument('--w_conv2',    type=float, default=0.05,        help='')
parser.add_argument('--w_conv3',    type=float, default=0.005,       help='')
parser.add_argument('--w_fc1',      type=float, default=0.001,      help='')
parser.add_argument('--w_fc2',      type=float, default=0.0001,     help='')

# Dataset path and specification
# --------------------------------------------------------------------------------------------------
parser.add_argument('--log', default='default.csv')
parser.add_argument('--log_train', default='log_train.csv')
parser.add_argument('-d', '--dataset', default='HHAR', help='e.g. HHAR, DAR')
parser.add_argument('-ss', '--subject_source',  default='d', help='e.g. a-i for HHAR; teacher, student1 for DAR')
parser.add_argument('-st', '--subject_target',  default='e', help='e.g. a-i for HHAR; teacher, student1 for DAR')
parser.add_argument('-ps', '--position_source', default='nexus4', help='None for HHAR, la, lw, ra, rw for DAR')
parser.add_argument('-pt', '--position_target', default='nexus4', help='None for HHAR, la, lw, ra, rw for DAR')
parser.add_argument('--scaling', type='bool', default=True)

p = parser.parse_args()

import logging

#setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to error
handler = logging.FileHandler('./datasetLog_'+p.dataset+ '/source_'+ p.subject_source+'_'+p.position_source+ ' target_'+ p.subject_target+ '_'+ p.position_target+' '+p.mode+'.log',"w", encoding=None, delay="true")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.debug('Current experiment parameters:')
pprint(vars(p))
logging.debug(vars(p))
# %%

sys.path.append('src')

torch.manual_seed(p.seed)
np.random.seed(p.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = p.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path_prefix = ''
if p.dataset == 'HHAR':
    data_path = '/content/drive/My Drive/Colab Notebooks/tcp_work/hdcnn-x-master/data/preprocessed/hhar_all_users.npz'
    with np.load(data_path, mmap_mode='r', allow_pickle=True) as npz:
        data_source = npz['data_{}_{}_train'.format(
            p.subject_source, p.position_source)]
        data_source_small = npz['data_{}_{}_test'.format(
            p.subject_source, p.position_source)]
        data_target = npz['data_{}_{}_train'.format(
            p.subject_target, p.position_target)]
        data_target_small = npz['data_{}_{}_test'.format(
            p.subject_target, p.position_target)]

elif p.dataset == 'PAMAP2':
    data_path = '/content/drive/My Drive/Colab Notebooks/tcp_work/hdcnn-x-master/data/preprocessed/pamap2_all_users.npz'
    with np.load(data_path, mmap_mode='r', allow_pickle=True) as npz:
        data_source = npz['data_{}_{}_train'.format(
            p.subject_source, p.position_source)]
        data_source_small = npz['data_{}_{}_test'.format(
            p.subject_source, p.position_source)]
        data_target = npz['data_{}_{}_train'.format(
            p.subject_target, p.position_target)]
        data_target_small = npz['data_{}_{}_test'.format(
            p.subject_target, p.position_target)]

elif p.dataset == 'OPPORTUNITY':
    data_path = 'data/preprocessed/opportunity_all_users.npz'
    with np.load(data_path, mmap_mode='r', allow_pickle=True) as npz:
        data_source = npz['data_{}_{}'.format(
            p.subject_source, p.position_source)]
        data_target = npz['data_{}_{}'.format(
            p.subject_target, p.position_catarget)]

#data_target_small = data_target[:1000, :]
hdcnn = HDCNN(device, data_source, data_target_small, p)
hdcnn.train_source(p)
hdcnn.train_target(p)

#torch.save(hdcnn, '/content/drive/My Drive/Colab Notebooks/tcp_work/')

hdcnn.test(hdcnn.net_FCt, DataLoader(ActivityDataset(data_target, p.window_size, p.n_channels, p.scaling),
                                     batch_size=p.batch_size, shuffle=True, drop_last=True), 'TARGET FULL SET:')