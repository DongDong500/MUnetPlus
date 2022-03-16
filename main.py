import os
import sys
import random
import argparse

from datetime import datetime
import timeit
import socket
import glob

import utils
import network
from datasets import CPNSegmentation
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils.dice_score import dice_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
from PIL import Image

from train import train

def get_argparser():
    parser = argparse.ArgumentParser()

    # Tensorboard Options
    parser.add_argument("--Tlog_dir", type=str, default='/home/DATA/sdi3/',
                        help="path to Tensorboard log")
    parser.add_argument("--save_log", action='store_true', default=False,
                        help='save tensorboard logs to \"/home/DATA/sdi\"')

    # Model Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='unet_rgb',
                        choices=available_models, help='model name')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='/data/sdi/MUnet/datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default="CPN", 
                        choices=['CPN', 'CPN_c', 'median', 'CTS', 'muscleUS'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num class (default: 2")
    parser.add_argument("--is_rgb", action='store_true', default=False)

    # Train Options
    parser.add_argument("--gpus", type=str, default='cpu')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    # Save best model checkpoint
    parser.add_argument("--save_model", action='store_true', default=False,
                        help='save best model param to \"./best_param\"')
    parser.add_argument("--save_ckpt", type=str, default='/home/DATA/sdi/',
                        help="save best model param to \"./best_param\"")

    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=(640, 512))
    parser.add_argument("--scale_factor", type=float, default=5e-1)

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'entropy_dice_loss', 'dice_loss'], 
                        help="loss type")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=5e-1,
                        help='weight decay (default: 5e-1)') 
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')

    # Validate Options
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 10)")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help='save segmentation results to \"./val_results\"')
    parser.add_argument("--save_val_dir", type=str, default='/home/DATA/sdi/',
                        help="save segmentation results to \"./results\"")

    # Save result images Options
    ctime = 'default'
    parser.add_argument("--save_train_results", action='store_true', default=False,
                        help='save segmentation results to \"./train_results\"')
    parser.add_argument("--save_train_dir", type=str, default='/home/DATA/sdi/',
                        help="save segmentation results to \"./train_results\"")
    parser.add_argument("--current_time", type=str, default=ctime,
                        help="results images folder name (default: current time)")

    return parser

def saveSummary(opts):
    
    with open(os.path.join(opts.Tlog_dir, opts.model, 
                    opts.current_time + '_' + socket.gethostname(), 'summary.txt'), 'w') as f:
            for k, v in vars(opts).items():
                f.write("{}={}\n".format(k, v))
            

if __name__ == '__main__':

    opts = get_argparser().parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    elapsed_times = []
    try:
        for loss_name in ['entropy_dice_loss', 'cross_entropy', 'dice_loss', 'focal_loss']:
            for lr in [5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]:

                opts.loss_type = loss_name
                if loss_name == 'focal_loss':
                    opts.lr = lr*1e+4
                elif loss_name == 'entropy_dice_loss':
                    opts.lr = lr*1e-3
                else:
                    opts.lr = lr

                opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                start_time = datetime.now()
                train(devices=device, opts=opts)
                time_elapsed = datetime.now() - start_time
                print('Time elapsed (h:m:s.ms) {}'.format(time_elapsed))
                elapsed_times.append(time_elapsed)
    except KeyboardInterrupt:
        print("Stop !!!")
    
    print(elapsed_times)