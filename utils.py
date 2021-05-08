import os
import torch 
from torchvision.utils import make_grid
import dataio
import matplotlib.cm


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
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
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img

def write_image_summary(prefix, gt, pred, image, writer, total_steps):
    gt_depth = gt[0].unsqueeze(0) # pick the first image in the batch
    pred_depth = pred[0].unsqueeze(0)
    ori_img = image[0].unsqueeze(0)
    pred_vs_gt = torch.cat((gt_depth, pred_depth), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(pred_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    ori_img = dataio.rescale_img((ori_img+1)/2, mode='clamp').permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    
    
    writer.add_image(prefix + 'ori_img', torch.from_numpy(ori_img).permute(2,0,1), global_step=total_steps)
    