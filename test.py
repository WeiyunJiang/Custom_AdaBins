import numpy as np
import torch
import torch.utils.data.distributed
import random
import os

import utils

from tqdm import tqdm

from models import VGG_16, UnetAdaptiveBins

from dataio import Depth_Dataset

from args import depth_arg
from torch.utils.data import DataLoader

from evaluate import evaluate_model, RunningAverageDict


def test(model, test_data_loader):
    with torch.no_grad():
        metrics_test = RunningAverageDict()
        model.eval()
        
        for step, batch in tqdm(enumerate(test_data_loader)):  
            # image(N, 3, 427, 565)
            # depth(N, 1, 427, 565)
            image, depth = batch['image'], batch['depth']
            image = image.to(device)
            depth = depth.to(device)
        
            bins, pred = model(image)
            evaluate_model(pred, depth, metrics_test, args)
        
        metrics_test_value = metrics_test.get_value()
        tqdm.write("SiLog Loss: %.4f, a1: %.4f, a2: %.4f, a3: %.4f, rel: %.4f, rms: %.4f, log10: %.4f" 
                % (metrics_test_value['silog'], metrics_test_value['a1'], metrics_test_value['a2'],
                   metrics_test_value['a3'],metrics_test_value['abs_rel'],metrics_test_value['rmse'],
                   metrics_test_value['log_10']))


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
    
    test_dataset = Depth_Dataset(args.dataset, 'test', small_data_num = None)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
    train_state_dict = torch.load(PATH)

    model = UnetAdaptiveBins.build_encoder(n_bins=args.n_bins, min_val=args.min_depth, 
                                           max_val=args.max_depth, norm=args.norm)
    model.load_state_dict(train_state_dict)
    model.to(device)
    test(model, test_data_loader)








