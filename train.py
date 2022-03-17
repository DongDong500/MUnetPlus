import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np

import os
import random
import socket
from tqdm import tqdm

from PIL import Image

import network
from utils import ext_transforms as et
'''
import utils
from datasets import CPNSegmentation
from metrics import StreamSegMetrics

from utils.dice_score import dice_loss
'''

def get_dataset(opts):
    if opts.is_rgb:
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        ...

def validate():
    ...

def save_validate_results():
    ...

def train(devices=None, opts=None):

    if devices is None or opts is None:
        raise Exception
    
    LOGDIR = os.path.join(opts.Tlog_dir, opts.model, opts.current_time+'_'+socket.gethostname())
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    # Tensorboard 
    if opts.save_log:
        logdir = os.path.join(LOGDIR, 'log')
        writer = SummaryWriter(log_dir=logdir)
    # Validate
    if opts.save_val_results:
        logdir = os.path.join(LOGDIR, 'val_results')
        os.mkdir(logdir)
        opts.save_val_dir = logdir
    # Train
    if opts.save_train_results:
        logdir = os.path.join(LOGDIR, 'train_results')
        os.mkdir(logdir)
        opts.save_train_dir = logdir
    # CheckPoint
    if opts.save_model:
        logdir = os.path.join(LOGDIR, 'best_param')
        os.mkdir(logdir)
        opts.save_ckpt = logdir
    # Save Options description
    with open(os.path.join(LOGDIR, 'summary.txt'), 'w') as f:
        for k, v in vars(opts).items():
            f.write("{} : {}\n".format(k, v))

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ''' (1) Get Datasets
    '''
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" % 
                    (opts.dataset, len(train_dst), len(val_dst)))

    ''' (2) Load Model
    '''
    try:
        print("Model selection: {}".format(opts.model))
        model = network.modeling.__dict__[opts.model](channel=3 if opts.is_rgb else 1, 
                                                        num_class=opts.num_classes)
    except:
        raise Exception
    






    

    