import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import random
import time
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import utils

from tqdm import tqdm

from models import VGG_16, UnetAdaptiveBins

from dataio import Depth_Dataset
from loss import SILogLoss, BinsChamferLoss
from args import depth_arg
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from evaluate import *

def validation(model, model_dir, val_data_loader, epoch, total_steps, best_val_abs_rel, args):
    with torch.no_grad():
        metrics_val = RunningAverageDict()
        model.eval()
        summaries_dir = os.path.join(model_dir, 'val_summaries')
        utils.cond_mkdir(summaries_dir)
    
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)
        
        writer = SummaryWriter(summaries_dir)
        for step, batch in enumerate(val_data_loader):  
            # image(N, 3, 427, 565)
            # depth(N, 1, 427, 565)
            
            
            image, depth = batch['image'], batch['depth']
            image = image.to(device)
            depth = depth.to(device)
            
            bins, pred = model(image)
            evaluate(pred, depth, metrics_val, args)
            
        metrics_val_value = metrics_val.get_value()
        writer.add_scalar("step_val_silog_loss", metrics_val_value['silog'], total_steps)
        writer.add_scalar("step_val_a1", metrics_val_value['a1'], total_steps)
        writer.add_scalar("step_val_a2", metrics_val_value['a2'], total_steps)
        writer.add_scalar("step_val_a3", metrics_val_value['a3'], total_steps)
        writer.add_scalar("step_val_rel", metrics_val_value['abs_rel'], total_steps)
        writer.add_scalar("step_val_rms", metrics_val_value['rmse'], total_steps)
        writer.add_scalar("step_val_log10", metrics_val_value['log_10'], total_steps)
        
        if  metrics_val_value['abs_rel'] < best_val_abs_rel:
            best_val_abs_rel = metrics_val_value['abs_rel']
            torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_best_val.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'best_psnr_epoch.txt'),
                    np.array([best_val_abs_rel, epoch]))
                
    
    
def train_model(model, model_dir, args, summary_fn=None, device=None):
    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    
    # initialize dataset
    train_dataset = Depth_Dataset(args.dataset, 'train', small_data_num = args.small_data_num)
    num_train = int(0.9 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                               [num_train, 
                                                                num_val])
   	
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # define loss criterion for depth and bin maps
    criterion_depth = SILogLoss() 
    criterion_bins = BinsChamferLoss()
    
    model.train(True)
    
    # we want to tune the parameter of the pretrained encoder more carefully
    params = [{"params": model.get_1x_lr_params(), "lr": args.lr / 10},
              {"params": model.get_10x_lr_params(), "lr": args.lr}]
    
    # define optimizer
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    
    # one cycle lr scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, 
                                              steps_per_epoch=len(train_data_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, 
                                              last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)

    total_steps = 0
    metrics = RunningAverageDict()
    best_train_abs_rel = 100
    best_val_abs_rel = 100
    
    with tqdm(total=len(train_data_loader) * args.epochs) as pbar:
        for epoch in range(args.epochs):
            print("Epoch {}/{}".format(epoch, args.epochs))
            print('-' * 10)
            epoch_train_losses = []
            epoch_train_silog_losses = []
            
            if not (epoch+1) % args.epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                
            for step, batch in enumerate(train_data_loader):
                start_time = time.time()
                
                # image(N, 3, 427, 565)
                # depth(N, 1, 427, 565)
                optimizer.zero_grad()
                
                image, depth = batch['image'], batch['depth']
                image = image.to(device)
                depth = depth.to(device)
                
                bins, pred = model(image)

                mask = depth > args.min_depth
                mask = mask.to(torch.bool)
                loss_depth = criterion_depth(pred, depth, mask=mask)
                loss_bin = criterion_bins(bins, depth)
                
                loss = loss_depth + args.w_chamfer * loss_bin
                loss.backward()
                
                epoch_train_losses.append(loss.clone().detach().cpu().numpy())
                epoch_train_silog_losses.append(loss_depth.clone().detach().cpu().numpy())
                
                clip_grad_norm_(model.parameters(), 0.1)  # optional
                optimizer.step()

                
                with torch.no_grad():
                    evaluate(pred, depth, metrics, args)
                    
                
                scheduler.step()
                
                pbar.update(1)
                
                if not (total_steps+1) % args.steps_til_summary:
                    tqdm.write("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, iteration time %0.6f sec" 
                    % (epoch, args.epochs, step, len(train_data_loader), loss, time.time() - start_time))
                    
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                    metrics_value = metrics.get_value()
                    writer.add_scalar("step_train_loss", loss, total_steps)
                    writer.add_scalar("step_train_silog_loss", metrics_value['silog'], total_steps)
                    writer.add_scalar("step_train_a1", metrics_value['a1'], total_steps)
                    writer.add_scalar("step_train_a2", metrics_value['a2'], total_steps)
                    writer.add_scalar("step_train_a3", metrics_value['a3'], total_steps)
                    writer.add_scalar("step_train_rel", metrics_value['abs_rel'], total_steps)
                    writer.add_scalar("step_train_rms", metrics_value['rmse'], total_steps)
                    writer.add_scalar("step_train_log10", metrics_value['log_10'], total_steps)

                    for param_group in optimizer.param_groups:
                        writer.add_scalar("epoch_train_lr", param_group['lr'], total_steps)
                    # summary_fn(depth, pred, image, writer, total_steps)
                        
                total_steps += 1
                
            writer.add_scalar("epoch_train_loss", np.mean(epoch_train_losses), epoch)
            writer.add_scalar("epoch_train_silog_loss", np.mean(epoch_train_silog_losses), epoch)
            
            ## validation
            validation(model, model_dir, val_data_loader, epoch, total_steps, best_val_abs_rel, args)


if __name__ == '__main__': 
    args = depth_arg()
    # Set random seed
    print(f'Using random')
    print(f'Using random seed {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    root_path = os.path.join(args.logging_root, args.exp_name)
    
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)    
    model = UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, min_val=args.min_depth, 
                                           max_val=args.max_depth, norm=args.norm)
    '''
    output_size = (320, 240) 
    model = VGG_16(output_size= output_size) 
    params = model.parameters()
    '''
    model.to(device) 
    args.epoch = 0 
    args.last_epoch = -1
    
    train_model(model, root_path, args, summary_fn=None, device=device)


























