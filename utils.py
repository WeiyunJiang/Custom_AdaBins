import os
import torch 
from torchvision.utils import make_grid
import dataio
import matplotlib.cm


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
    colored_img_scaled = float(colored_img) / 255
    #     return img.transpose((2, 0, 1))
    return colored_img_scaled # (H, W, 3)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_image_summary(prefix, gt, pred, image, writer, total_steps):
    # pred (H,W, 3) gt (H, W, 3) image(3, H, W)
    gt_depth = gt.permute(2,0,1) #(3, H, W)
    pred_depth = pred.permute(2,0,1) #(3, H, W)
    ori_img = image #(3, H, W)
    pred_vs_gt = torch.cat((gt_depth, pred_depth), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(pred_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    ori_img = dataio.rescale_img((ori_img+1)/2, mode='clamp').detach().cpu().numpy()
    
    
    writer.add_image(prefix + 'ori_img', torch.from_numpy(ori_img), global_step=total_steps)
    