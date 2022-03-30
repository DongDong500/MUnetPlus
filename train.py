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

import utils
import network
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et


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
        raise NotImplementedError

    if opts.dataset == "CPN":
        train_dst = dt.CPNSegmentation(root=opts.data_root, datatype='CPN', image_set='train',
                                     transform=train_transform, is_rgb=False)
        val_dst = dt.CPNSegmentation(root=opts.data_root, datatype='CPN', image_set='val',
                                  transform=val_transform, is_rgb=False)
    else:
        raise NotImplementedError
    
    return train_dst, val_dst


def save_val_image(opts, model, loader, device, epoch):

    if not os.path.exists(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch))):
        try:
            os.mkdir(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch)))
        except:
            raise Exception
    save_dir = os.path.join(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch)))

    if opts.is_rgb:
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError

    for i, (images, labels) in tqdm(enumerate(loader)):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        probs = nn.Softmax(dim=1)(outputs)

        preds = torch.max(probs, 1)[1].detach().cpu().numpy()
        image = images.detach().cpu().numpy()
        lbl = labels.detach().cpu().numpy()

        for j in range(images.shape[0]):
            tar1 = (denorm(image[i]) * 255).transpose(1, 2, 0).astype(np.uint8)
            tar2 = (lbl[i] * 255).astype(np.uint8)
            tar3 = (preds[i] * 255).astype(np.uint8)
            Image.fromarray(tar1).save(os.path.join(save_dir, '{}_image.png'.format(i*images.shape[0] + j)))
            Image.fromarray(tar2).save(os.path.join(save_dir, '{}_mask.png'.format(i*images.shape[0] + j)))
            Image.fromarray(tar3).save(os.path.join(save_dir, '{}_preds.png'.format(i*images.shape[0] + j)))
    

def validate(opts, model, loader, device, metrics, epoch, criterion):

    metrics.reset()
    ret_samples = []

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()
            
            metrics.update(target, preds)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

        if opts.save_val_results:
            save_val_image(opts, model, loader, device, epoch)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss  


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
    # Check Point
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
        # Data Parallel Option
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    except:
        raise Exception
    
    ''' (2-1) Resume model
    '''
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        raise NotImplementedError
    else:
        print("Train from scratch...")
        resume_epoch = 0
        model.to(devices)

    ''' (3) Set up criterion
    '''
    if opts.loss_type == 'cross_entropy':
        criterion = utils.CrossEntropyLoss(weight=torch.tensor([0.02, 0.98]).to(devices))
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(gamma=2, alpha=torch.tensor([0.02, 0.98]).to(devices))
    elif opts.loss_type == 'dice_loss':
        criterion = utils.DiceLoss()
    elif opts.loss_type == 'entropy_dice_loss':
        criterion = utils.EntropyDiceLoss(weight=torch.tensor([0.02, 0.98]).to(devices))
    else:
        raise NotImplementedError

    ''' (4) Set up optimizer
    '''
    optimizer = optim.RMSprop(model.parameters(), 
                                lr=opts.lr, 
                                weight_decay=opts.weight_decay,
                                momentum=opts.momentum)
    if opts.lr_policy == 'poly':
        raise NotImplementedError
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    ''' (5) Set up metrics
    '''
    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=40, verbose=True, 
                                            path=opts.save_ckpt, save_model=opts.save_model)
    best_score = 0.0

    ''' (6) Train
    '''
    for epoch in range(resume_epoch, opts.total_itrs):

        model.train()
        running_loss = 0.0
        metrics.reset()

        for (images, lbl) in tqdm(train_loader):

            images = images.to(devices)
            lbl = lbl.to(devices)
            
            optimizer.zero_grad()

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            metrics.update(lbl.detach().cpu().numpy(), preds)
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        score = metrics.get_results()

        epoch_loss = running_loss / len(train_loader.dataset)
        print("[{}] Epoch: {}/{} Loss: {}".format(
            'Train', epoch+1, opts.total_itrs, epoch_loss))
        print("Overall Acc: {:.2f}, Mean Acc: {:.2f}, FreqW Acc: {:.2f}, Mean IoU: {:.2f}, Class IoU [0]: {:.2f} [1]: {:.2f}".format(
            score['Overall Acc'], score['Mean Acc'], score['FreqW Acc'], score['Mean IoU'], score['Class IoU'][0], score['Class IoU'][1]))
        print("F1 [0]: {:.2f} [1]: {:.2f}".format(score['Class F1'][0], score['Class F1'][1]))
        
        if opts.save_log:
            writer.add_scalar('Overall_Acc/train', score['Overall Acc'], epoch)
            writer.add_scalar('Mean_Acc/train', score['Mean Acc'], epoch)
            writer.add_scalar('FreqW_Acc/train', score['FreqW Acc'], epoch)
            writer.add_scalar('Mean_IoU/train', score['Mean IoU'], epoch)
            writer.add_scalar('Class_IoU_0/train', score['Class IoU'][0], epoch)
            writer.add_scalar('Class_IoU_1/train', score['Class IoU'][1], epoch)
            writer.add_scalar('Class_F1_0/train', score['Class F1'][0], epoch)
            writer.add_scalar('Class_F1_1/train', score['Class F1'][1], epoch)
            writer.add_scalar('epoch_loss/train', epoch_loss, epoch)
        
        if (epoch+1) % opts.val_interval == 0:
                print("validation...")
                model.eval()
                metrics.reset()
                val_score, val_loss = validate(opts, model, val_loader, 
                                                devices, metrics, epoch, criterion)
                early_stopping(val_loss, model)

                print("[{}] Epoch: {}/{} Loss: {}".format('Validate', epoch+1, opts.total_itrs, val_loss))
                print("Overall Acc: {:.2f}, Mean Acc: {:.2f}, FreqW Acc: {:.2f}, Mean IoU: {:.2f}".format(
                    val_score['Overall Acc'], val_score['Mean Acc'], val_score['FreqW Acc'], val_score['Mean IoU']))
                print("Class IoU [0]: {:.2f} [1]: {:.2f}".format(val_score['Class IoU'][0], val_score['Class IoU'][1]))
                print("F1 [0]: {:.2f} [1]: {:.2f}".format(val_score['Class F1'][0], val_score['Class F1'][1]))
                
                if opts.save_log:
                    writer.add_scalar('Overall_Acc/val', val_score['Overall Acc'], epoch)
                    writer.add_scalar('Mean_Acc/val', val_score['Mean Acc'], epoch)
                    writer.add_scalar('FreqW_Acc/val', val_score['FreqW Acc'], epoch)
                    writer.add_scalar('Mean_IoU/val', val_score['Mean IoU'], epoch)
                    writer.add_scalar('Class_IoU_0/val', val_score['Class IoU'][0], epoch)
                    writer.add_scalar('Class_IoU_1/val', val_score['Class IoU'][1], epoch)
                    writer.add_scalar('Class_F1_0/val', val_score['Class F1'][0], epoch)
                    writer.add_scalar('Class_F1_1/val', val_score['Class F1'][1], epoch)
                    writer.add_scalar('epoch_loss/val', val_loss, epoch)
        
        if early_stopping.early_stop:
            print("Early Stop !!!")
            if opts.save_val_results:
                save_val_image(opts, model, val_loader, devices, epoch)
            break


    

    