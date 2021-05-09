import matplotlib.pyplot as plt
from time import time
from PIL import Image
from models import UnetAdaptiveBins
from torchvision.transforms import ToTensor
import os 
from args import depth_arg
import torch
import numpy as np
import utils

class InferenceHelper:
    def __init__(self, args, device, dataset='nyu'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            
        

        model = model.load_state_dict(train_state_dict) 
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

if __name__ == '__main__':
    args = depth_arg()
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)  
    
    img = Image.open("dataset/nyu_depth_v2/official_splits/test/classroom/rgb_00283.jpg")
    start = time()
    root_path = os.path.join(args.logging_root, args.exp_name)
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
    train_state_dict = torch.load(PATH)
    inferHelper = InferenceHelper(args, device)
    centers, pred = inferHelper.predict_pil(img)
    print(f"took :{time() - start}s")
    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.show()
    plt.savefig('test_imgs/classroom__rgb_00283_depth.jpg')