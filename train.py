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

from models import VGG_16, UnetAdaptiveBins, VGG_UnetAdaptiveBins, UnetSwinAdaptiveBins
import dataio
from dataio import Depth_Dataset
from loss import SILogLoss, BinsChamferLoss, MSELoss, BerhuLoss
from args import depth_arg
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from evaluate import evaluate_model, RunningAverageDict

def validation(model, optimizer, model_dir, val_data_loader, epoch, total_steps, best_val_abs_rel, args):
    with torch.no_grad():
        metrics_val = RunningAverageDict()
        model.eval()
        summaries_dir = os.path.join(model_dir, 'val_summaries')
        utils.cond_mkdir(summaries_dir)
    
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)
        
        writer = SummaryWriter(summaries_dir)
        depth_gt_list = []
        pred_list = []
        for step, batch in tqdm(enumerate(val_data_loader)):  
            # image(N, 3, 240, 320)
            # depth(N, 1, 240, 320)
            
            
            image, depth = batch['image'], batch['depth']
            image = image.to(device)
            depth = depth.to(device)
            
            bins, pred = model(image)
            evaluate_model(pred, depth, metrics_val, args)
            depth_gt_list.append(depth[0]) # (1, H, W)
            pred_list.append(pred[0])
            
        depth_gt = depth_gt_list[0] # (1, H, W)
        pred = pred_list[0]
        depth_gt[depth_gt < args.min_depth] = args.min_depth
        depth_gt[depth_gt > args.max_depth] = args.max_depth
        # gt_rescaled = dataio.rescale_img(depth_gt, mode='scale')
        # pred_rescaled = dataio.rescale_img(pred, mode='scale')
        colored_gt = utils.colorize(depth_gt, vmin=None, vmax=None, cmap='magma_r') # (H, W, 3)
        colored_pred = utils.colorize(pred, vmin=None, vmax=None, cmap='magma_r') # (H, W, 3)
        bins = bins[0]
        bins = bins.cpu().squeeze().numpy()
        bins = bins[bins > args.min_depth]
        bins = bins[bins < args.max_depth]
        
        utils.write_image_summary('val_', colored_gt, colored_pred, 
                                  image[0], depth_gt, bins, writer, total_steps)
        
                    
        metrics_val_value = metrics_val.get_value()
        writer.add_scalar("step_val_silog_loss", metrics_val_value['silog'], total_steps)
        writer.add_scalar("step_val_a1", metrics_val_value['a1'], total_steps)
        writer.add_scalar("step_val_a2", metrics_val_value['a2'], total_steps)
        writer.add_scalar("step_val_a3", metrics_val_value['a3'], total_steps)
        writer.add_scalar("step_val_rel", metrics_val_value['abs_rel'], total_steps)
        writer.add_scalar("step_val_rms", metrics_val_value['rmse'], total_steps)
        writer.add_scalar("step_val_log10", metrics_val_value['log_10'], total_steps)

        tqdm.write("Val SiLog Loss: %.4f, a1: %.4f, a2: %.4f, a3: %.4f, rel: %.4f, rms: %.4f, log10: %.4f" 
                   % (metrics_val_value['silog'], metrics_val_value['a1'], metrics_val_value['a2'],
                      metrics_val_value['a3'],metrics_val_value['abs_rel'],metrics_val_value['rmse'],
                      metrics_val_value['log_10']))
        
        if metrics_val_value['abs_rel'] < best_val_abs_rel:
            best_val_abs_rel = metrics_val_value['abs_rel']
            checkpoint = {
                'epoch': epoch + 1,
                'total_steps': total_steps + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint,
                       os.path.join(checkpoints_dir, 'model_best_val.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'best_psnr_epoch.txt'),
                       np.array([best_val_abs_rel, epoch]))
                
    
    
def train_model(model, model_dir, args, summary_fn=None, device=None):
    if args.berhuloss is True:
        print("Using new Berhu loss function" + "!" * 100)
    if args.resume is True:
        summaries_dir = os.path.join(model_dir, 'summaries')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        print(f'Resume Training from epoch {start_epoch}')
        model.load_state_dict(checkpoint['state_dict'])
        print('model loaded')
    
        
    else:
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
    train_dataset = Depth_Dataset(args.dataset, 'train', data_aug=args.data_aug,
                                  small_data_num = args.small_data_num)
    num_train = int(0.9 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                               [num_train, 
                                                                num_val])
   	
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # define loss criterion for depth and bin maps
    # criterion_depth = MSELoss()
    criterion_depth = SILogLoss()
    criterion_new = BerhuLoss()
    criterion_bins = BinsChamferLoss()
    
    model.train(True)
    
    # we want to tune the parameter of the pretrained encoder more carefully
    if args.pretrain is True:
        print('Using different learning rates')
        params = [{"params": model.get_1x_lr_params(), "lr": args.lr / 10},
                  {"params": model.get_10x_lr_params(), "lr": args.lr}]
    else:
        print('Using same learning rate')
        params = model.parameters()
        
    
    # define optimizer
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if args.resume is True:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('optimizer loaded')
    # one cycle lr scheduler
    '''
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, 
                                              steps_per_epoch=len(train_data_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, 
                                              last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    '''
    if args.resume is True:
        total_steps = checkpoint['total_steps']
        print(f'total steps start from {total_steps}')
    else:
        total_steps = 0
        print(f'total steps start from {total_steps}')
    metrics = RunningAverageDict()
    best_train_abs_rel = 100
    best_val_abs_rel = 100
    
    with tqdm(total=len(train_data_loader) * args.epochs) as pbar:
        for epoch in range(args.epochs):
            if args.resume is True:
                epoch += start_epoch
                total_epochs = args.epochs + start_epoch
            else:
                total_epochs = args.epochs
            print("Epoch {}/{}".format(epoch, total_epochs))
            print('-' * 10)
            epoch_train_losses = []
            epoch_train_SILOG_losses = []
            
            if not (epoch+1) % args.epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                
            for step, batch in enumerate(train_data_loader):
                start_time = time.time()
                
                # image(N, 3, 240, 320)
                # depth(N, 1, 240, 320)
                optimizer.zero_grad()
                
                image, depth = batch['image'], batch['depth']
                image = image.to(device)
                depth = depth.to(device)
                
                bins, pred = model(image)
                
                mask = depth > args.min_depth
                mask = mask.to(torch.bool)
                loss_depth = criterion_depth(pred, depth, mask=mask)
                
                loss_bin = criterion_bins(bins, depth)
                if args.berhuloss_only is not True:
                    #print('Use complete silog loss')
                    loss = loss_depth + args.w_chamfer * loss_bin
                else:
                    #print("Use complete berhu loss")
                    loss = criterion_new(pred, depth, mask=mask) + args.w_chamfer * loss_bin
                if args.berhuloss is True:
                    #print('Use partial berhu loss')
                    loss_new = criterion_new(pred, depth, mask=mask)
                    loss = loss + loss_new * 20
                
 
                loss.backward()
                
                epoch_train_losses.append(loss.clone().detach().cpu().numpy())
                epoch_train_SILOG_losses.append(loss_depth.clone().detach().cpu().numpy())
                
                clip_grad_norm_(model.parameters(), 0.1)  # optional
                optimizer.step()

                
                with torch.no_grad():
                    evaluate_model(pred, depth, metrics, args)
                    depth_gt = depth[0] # (1, H, W)
                    depth_gt[depth_gt < args.min_depth] = args.min_depth
                    depth_gt[depth_gt > args.max_depth] = args.max_depth
                    pred = pred[0] # (1, H, W)
                    #pred_up = nn.functional.interpolate(pred, depth_gt.shape[-2:], mode = 'bilinear', align_corners=True)
                    
                    
                    colored_gt = utils.colorize(depth_gt, vmin=None, vmax=None, cmap='magma_r') # (H, W, 3)
                    colored_pred = utils.colorize(pred, vmin=None, vmax=None, cmap='magma_r') # (H, W, 3)
                    # pred_rescaled = dataio.rescale_img(pred, mode='scale')
                    # gt_rescaled = dataio.rescale_img(depth_gt, mode='scale')
                    bins = bins.cpu().squeeze().numpy()
                    bins = bins[0]
                    bins = bins[bins > args.min_depth]
                    bins = bins[bins < args.max_depth]
                    
                    
                    utils.write_image_summary('train_', colored_gt, colored_pred, 
                                              image[0], depth_gt, bins, writer, total_steps)
                
                #scheduler.step()
                
                pbar.update(1)
                
                if not (total_steps+1) % args.steps_til_summary:
                    tqdm.write("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, iteration time %0.6f sec" 
                    % (epoch, total_epochs, step, len(train_data_loader), loss, time.time() - start_time))
                    
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
            writer.add_scalar("epoch_train_SILOG_loss", np.mean(epoch_train_SILOG_losses), epoch)
            
            ## validation
            validation(model, optimizer, model_dir, val_data_loader, epoch, total_steps, best_val_abs_rel, args)


if __name__ == '__main__': 
    args = depth_arg()
    # Set random seed
   
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
    if args.name == 'UnetAdaptiveBins':
        model = UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, 
                                               pretrained=args.pretrain, 
                                               min_val=args.min_depth, 
                                               max_val=args.max_depth, 
                                               norm=args.norm)
    elif args.name == 'VGG_UnetAdaptiveBins':
        model = VGG_UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, 
                                                   pretrained=args.pretrain,
                                                   min_val=args.min_depth, 
                                                   max_val=args.max_depth, 
                                                   norm=args.norm)
    elif args.name == 'UnetSwinAdaptiveBins':
        model = UnetSwinAdaptiveBins.build_encoder(n_bins=args.n_bins, 
                                               pretrained=args.pretrain, 
                                               min_val=args.min_depth, 
                                               max_val=args.max_depth, 
                                               norm=args.norm)
    else:
        raise NotImplementedError('Not implemented for name={args.name}')
    
    '''
    output_size = (320, 240) 
    model = VGG_16(output_size= output_size) 
    params = model.parameters()
    '''
    
    model.to(device) 
    total_n_params = utils.count_parameters(model)
    print(f'Total number of parameters of {args.name}: {total_n_params}')
    
    #args.epoch = 0 
    args.last_epoch = -1
    
    train_model(model, root_path, args, summary_fn=None, device=device)


























