# Custom_AdaBins
Custom_AdaBins (reimplementation of AdaBins: https://github.com/shariqfarooq123/AdaBins)
Added Features
- VGG-16 encoder
- Swin transformer AdaBins block
- New Berhu loss function
- Online data augmentation
- Transfer learning of encoder
## Prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set

- `mkdir dataset`
- `cd dataset`
- `mkdir nyu_depth_v2`
- `cd nyu_depth_v2`
- `mkdir official_splits`
- `gdown https://drive.google.com/uc?id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP` [1] uniformly sample frames from the entire training scenes and extract approximately 24K unique pairs
[1] Heo, Minhyeok, et al, "Monocular depth estimation using whole strip masking and reliability-based refinement." Proceedings of the European Conference on Computer Vision (ECCV). 2018. 
### Get official NYU Depth V2 split file
- `wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat`
### Convert mat file to image files
- `python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./dataset/nyu_depth_v2/official_splits/`

## effnet_mini_ViT (baseline)
**Train**
- `python train.py --exp_name effnet_mini_ViT --name UnetAdaptiveBins --data_aug False --epochs 150 --batch_size 10`
 
**Test**
- `python test.py --exp_name effnet_mini_ViT --name UnetAdaptiveBins`
## vgg16_mini_ViT 
**Train**
- `python train.py --exp_name vgg_mini_ViT --name VGG_UnetAdaptiveBins --data_aug False --epochs 150 --batch_size 10` 

**Test**
- `python test.py --exp_name vgg_mini_ViT --name VGG_UnetAdaptiveBins`

## vgg16_mini_Swin
**Train**
- `python train.py --exp_name effnet_mini_Swin --name UnetSwinAdaptiveBins --data_aug False --epochs 150 --batch_size 10` 

**Test**
- `python test.py --exp_name effnet_mini_Swin --name UnetSwinAdaptiveBins `

##Demo##
python demo.py --exp_name effnet_mini_ViT --name UnetAdaptiveBins
