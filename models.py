#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:02:59 2021

@author: weiyunjiang
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class VGG_16(nn.Module):
    """ Naive VGG-16
    
        
    """
    def __init__(self, output_size=(320, 240)):
        super(VGG_16, self).__init__()
        self.output_size = output_size
        self.vgg = vgg16_bn(pretrained=True)
        self.vgg.classifier._modules['6'] = nn.Linear(4096, output_size[0]*output_size[1])
        self.transform = torch.nn.functional.interpolate

        
    def forward(self, image):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.vgg(image)
        out = F.relu(out)
        out += 1e-3
        out = out.view(-1, *self.output_size)
        return out