#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:16:03 2021

@author: weiyunjiang
"""
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# from torchvision.transforms import Normalize, RandomGrayscale, Compose, ToTensor, ColorJitter, RandomHorizontalFlip, RandomApply, RandomRotation


class Depth_Dataset(Dataset):
    """
    data_name: {nyu}
    split: {train, val, test}
    small_data_num: number of data samples used
    
    """
    def __init__(self, data_name, split, data_aug=False, resolution=(320, 240), 
                 downsample=True, small_data_num=None):
        self.downsample = downsample
        self.resolution = resolution
        if data_name == 'nyu':
            if split == 'train':
                with open("./train_test_inputs/official_nyudepthv2_train_files_with_gt.txt", 'r') as f:
                    self.filenames = f.readlines()
                    self.data_pth = './dataset/nyu_depth_v2/official_splits/'
            elif split == 'test':
                with open("./train_test_inputs/nyudepthv2_test_files_with_gt.txt", 'r') as f:
                    self.filenames = f.readlines()
                    self.data_pth = './dataset/nyu_depth_v2/official_splits/test/'
            else:
                raise NotImplementedError('Not implemented for split={split}')
                
        else:
            raise NotImplementedError('Not implemented for data_name={data_name}')
          
        if data_aug is True:
            print('Using Data Augmentation')
            # brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            
            self.cjitter = color_jitter # only image
            
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform_depth = transforms.Compose([
                transforms.ToTensor(),
                    ])
        else:
            print('NOT using Data Augmentation !!!!')
            
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform_depth = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.split = split
        self.data_aug = data_aug
        self.small_data_num = small_data_num
        self.data_name = data_name
        if self.split == 'test':
            self.do_kb_crop = True # used for evaluation only
        else:
            self.do_kb_crop = False
    
    def __len__(self):
        if self.small_data_num == None:
            return len(self.filenames)
        else:
            return self.small_data_num
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        if self.data_name == 'nyu':
            if self.split == 'train':
                image_path = self.data_pth + sample_path.split()[0]
                depth_path = self.data_pth + sample_path.split()[1]
            elif self.split == 'test':
                image_path = self.data_pth + sample_path.split()[0]
                depth_path = self.data_pth + sample_path.split()[1]

        image = Image.open(image_path)
        depth_gt = Image.open(depth_path)
        image = image.crop((43, 45, 608, 472))
        depth_gt = depth_gt.crop((43, 45, 608, 472))
        if self.downsample:
            image = image.resize(self.resolution)
            depth_gt = depth_gt.resize(self.resolution)
            
        '''
        if self.do_kb_crop == True:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        '''
        # To avoid blank boundaries due to pixel registration
        # if self.data_name == 'nyu':
        #     depth_gt = depth_gt.crop((43, 45, 608, 472))
        #     image = image.crop((43, 45, 608, 472))
        if self.data_aug is True:
            # hflip with p = 0.5
            if random.random() > 0.5:
                image = TF.hflip(image)
                depth_gt = TF.hflip(depth_gt)
                
            # colorjitter with p = 0.2
            if random.random() > 0.8:
                image = self.cjitter(image)
                
            # (rotate -2.5 to 2.5 degrees with p=0.2)
            if random.random() > 0.8:
                angle_bound = 2.5
                random_angle = (random.random() - 0.5) * 2 * angle_bound
                image = image.rotate(random_angle)
                depth_gt = depth_gt.rotate(random_angle)
                
        image = np.asarray(image, dtype=np.uint8) 
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        if self.data_name == 'nyu':
            depth_gt = depth_gt / 1000.0
        elif self.data_name == 'kitti':
            depth_gt = depth_gt / 256.0
        else:
            raise NotImplementedError('Not implemented for data_name={data_name}')
        
        image = self.transform_image(image.copy())
        depth_gt = self.transform_depth(depth_gt.copy())
        sample = {'image': image, 'depth': depth_gt}
        
        return sample

def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x

if __name__ == '__main__':
    from models import VGG_16
    from args import depth_arg

    args = depth_arg()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_dataset = Depth_Dataset('nyu', 'train', data_aug=True, small_data_num=100)
    print(train_dataset[0])
    # train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [90, 10])
    # test
    test_dataset = Depth_Dataset('nyu', 'test')
    
    train_data_loader = DataLoader(train_dataset, batch_size=90, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    output_size = (320, 240)
    for i, batch in enumerate(train_data_loader):
        print(i)
        print(batch['image'].shape) # 1, 3, 427, 565
        print(batch['depth'].shape) # 1, 1, 427, 565
        depth = batch['depth']
        depth_rescaled = rescale_img(depth)
        depth_rescaled = depth_rescaled.numpy()[0].transpose(1,2,0)
        plt.imshow(depth_rescaled)
        
        image = batch['image']
        model = VGG_16(output_size=output_size) 
        out = model(image)
        print(out.shape) # 1, 320, 240
    pass
