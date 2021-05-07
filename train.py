import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

from models import VGG_16

from dataio import Depth_Dataset
from loss import SILogLoss
from args import get_train_args, depth_arg
from torch.utils.data import Dataset, DataLoader

'''
def main(args):

	
	# choose device
	gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
	

	
    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
	

	# build the dataset
	dataset = Depth_Dataset(args.dataset, args.split)
	if args.split == 'train':
		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset)-int(0.9 * len(train_dataset))]) 

    # build the model 
    output_size = (320, 40)
    model = VGG_16(output_size= output_size)
    

    # need optimizer
    params = model.parameters
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
'''

def train_model(model, lossF, optimizer, args, num_epochs=10):
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    model.to(device)

    
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    dataset = Depth_Dataset(args.dataset, args.split)
    if args.split == 'train':
    	train_dataset = Depth_Dataset('nyu', 'train', small_data_num = 100)
    	train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset)-int(0.9 * len(train_dataset))])
    	train_data_loader = DataLoader(train_dataset, batch_size=90, shuffle=True)
    	val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        lossF = lossF()
        for i, data in enumerate(train_dataset):
            '''
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
             
           	'''

            # inputs(3, 427, 565)
            # labels(1, 427, 565)

            inputs, labels = data['image'], data['depth']
            inputs.to(device)
            labels.to(device)

            

            a = inputs.size()
            b = labels.size()
            
            # inputs(1, 3, 427, 565)
            # labels(1, 1, 427, 565)
            inputs = inputs.view(1, a[0], a[1], a[2])
            labels = labels.view(1, b[0], b[1], b[2])
            
           
            
            
            optimizer.zero_grad()
            
            
            # pred 1,1,320,40
            pred = model(inputs)
            pred = pred.view(1, 1, 320, 240)
            
            
            
            mask = labels > args.min_depth
            

            
            loss = lossF(pred, labels, mask=mask.to(torch.bool))
            loss.backward()
            print(loss)
            
            
            optimizer.step()
    return loss

if __name__ == '__main__':
	args = depth_arg()
	output_size = (320, 240)
	model = VGG_16(output_size= output_size)
	params = model.parameters()

	#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
	print(train_model(model, SILogLoss, optimizer, args, num_epochs=10))


























