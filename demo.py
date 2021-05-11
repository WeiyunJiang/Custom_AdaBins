import matplotlib.pyplot as plt
from time import time
from PIL import Image
from models import UnetAdaptiveBins, VGG_UnetAdaptiveBins
from torchvision.transforms import ToTensor
import os 
from args import depth_arg
import torch
import numpy as np
import utils
import torch.nn as nn

class InferenceHelper:
    def __init__(self, train_state_dict, args, device, dataset='nyu'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            if args.name == 'UnetAdaptiveBins':
                self.model = UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, min_val=args.min_depth, 
                                                       max_val=args.max_depth, norm=args.norm)
            elif args.name == 'VGG_UnetAdaptiveBins':
                self.model = VGG_UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, min_val=args.min_depth, 
                                                       max_val=args.max_depth, norm=args.norm)
            else:
                raise NotImplementedError('Not implemented for name={args.name}')
            
        self.model.load_state_dict(train_state_dict) 
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            print('visualized')
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred
    
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final
if __name__ == '__main__':
    args = depth_arg()
    resolution = (320, 240)
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)  
    
    img = Image.open("dataset/nyu_depth_v2/official_splits/test/bathroom/rgb_00045.jpg")
    img = img.resize((320, 240))
    start = time()
    root_path = os.path.join(args.logging_root, args.exp_name)
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
    train_state_dict = torch.load(PATH)
    inferHelper = InferenceHelper(train_state_dict, args, device)
    centers, pred = inferHelper.predict_pil(img)
    print(f"took :{time() - start}s")
    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.axis('off')
    plt.show()
    plt.savefig(f'test_imgs/{args.exp_name}_bathroom__rgb_00045_depth.jpg', bbox_inches='tight')
