#!/bin/bash 

# HHAR
# ================================================================================================================
# HDCNN 77.79, train:  77.61,  prune, 79.%
python src/hdcnn_x.py -d 'HHAR' -ss 'a' -st 'c' -ps 'nexus4' -pt 'nexus4' -ch 3 -cls 6 -bs 32 -ws 128 -es 1 -et 20 --seed 2 --gpu 1 --w_conv1 0.005 --w_conv2 0.0002 --w_conv3 0.00001 --lr_FCt 0.0002 --log_interval 50 --mode prune
