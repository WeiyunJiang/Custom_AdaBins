# Custom_AdaBins
Custom_AdaBins
## Prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set

- `mkdir dataset`
- `cd dataset`
- `mkdir nyu_depth_v2`
$ cd nyu_depth_v2
$ mkdir official_splits
$ gdown https://drive.google.com/uc?id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP
[1] uniformly sample frames from the entire training scenes and extract approximately 24K unique pairs
[1] Heo, Minhyeok, et al, "Monocular depth estimation using whole strip masking and reliability-based refinement." Proceedings of the European Conference on Computer Vision (ECCV). 2018. 
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./dataset/nyu_depth_v2/official_splits/

