import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
#from .splits import split_dataset
'''
if __package__ is None:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from splits import split_dataset
else:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from .splits import split_dataset
'''
CpnDataDir = 'CPN_all'

class CPN(data.Dataset):
    """
    Args:6
        root (string): Root directory of the VOC Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, datatype='CPN_six', image_set='train', transform=None, is_rgb=True):
        
        is_aug = False

        self.root = os.path.expanduser(root)
        self.datafolder = datatype
        self.image_set = image_set
        self.transform = transform
        self.is_rgb = is_rgb

        cpn_root = os.path.join(self.root, datatype)
        image_dir = os.path.join(self.root, CpnDataDir, 'Images')
        mask_dir = os.path.join(self.root, CpnDataDir,'Masks')

        if not os.path.exists(cpn_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            raise NotImplementedError
        else:
            splits_dir = os.path.join(cpn_root, 'splits')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(splits_dir):
            split_dataset(splits_dir=splits_dir, data_dir=image_dir)

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered!' 
                             'Please use image_set="train" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError
        
        if self.is_rgb:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')         
        else:
            img = Image.open(self.images[index]).convert('L')
            target = Image.open(self.masks[index]).convert('L')            

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from splits import split_dataset

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=0.485, std=0.229)
            ])
    
    dlist = ['CPN_six']

    for j in dlist:
            
        dst = CPN(root='/data1/sdi/datasets', datatype=j, image_set='val',
                                    transform=transform, is_rgb=True)
        train_loader = DataLoader(dst, batch_size=1,
                                    shuffle=True, num_workers=2, drop_last=True)
        
        for i, (ims, lbls) in tqdm(enumerate(train_loader)):
            print(ims.shape)
            print(lbls.shape)
            print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            if i > 1:
                break
        