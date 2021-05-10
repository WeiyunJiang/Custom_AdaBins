import os
import torch 
from torchvision.utils import make_grid
import dataio
import matplotlib.cm
import numpy as np
import torch.nn.functional as F

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def colorize(value, vmin=None, vmax=None, cmap='magma_r'):
    value = value.detach().cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    colored_img = value[:, :, :3]
    colored_img_scaled = colored_img.astype(np.float) / 255
    #     return img.transpose((2, 0, 1))
    return colored_img_scaled # (H, W, 3)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_image_summary(prefix, gt, pred, image, depth_gt, bins, writer, total_steps):
    # pred (1, H, W) gt (1, H, W) image(3, H, W) if scaled
    # pred (H, W, 3) gt ( H, W, 3) image(3, H, W) if colorized
    pred = pred.transpose(2, 0, 1)
    gt = gt.transpose(2, 0, 1)
    gt_depth = gt.unsqueeze(0) #(1, 1, H, W)
    pred_depth = pred.unsqueeze(0) #(1, 1, H, W)
    pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:], mode='bilinear', align_corners=True)
    ori_img = image #(3, H, W)
    pred_vs_gt = torch.cat((gt_depth.squeeze(0), pred_depth.squeeze(0)), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(pred_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    ori_img = dataio.rescale_img(ori_img, mode='scale').detach().cpu().numpy()
    
    
    writer.add_image(prefix + 'ori_img', torch.from_numpy(ori_img), global_step=total_steps)
    depth_gt = depth_gt.squeeze()
    bins = bins.squeeze()
    writer.add_histogram(prefix + 'gt_bins', depth_gt, bins=30, global_step=total_steps)
    writer.add_histogram(prefix + 'pred_bins', bins, bins=30, global_step=total_steps)
    
    
    
