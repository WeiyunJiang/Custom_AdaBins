# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from swin_transformer import SwinTransformer
import torch
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def build_model():
    model_type = 'swin'
    if model_type == 'swin':
        # model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
        #                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                         in_chans=config.MODEL.SWIN.IN_CHANS,
        #                         num_classes=config.MODEL.NUM_CLASSES,
        #                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                         depths=config.MODEL.SWIN.DEPTHS,
        #                         num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                         qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         ape=config.MODEL.SWIN.APE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        model = SwinTransformer(img_size=(120, 160),
                                patch_size=4,
                                in_chans=128,
                                num_classes=10,
                                embed_dim=192,
                                depths=[6, 12],
                                num_heads=[6, 12],
                                window_size=5,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model 

if __name__ == '__main__':
    model = build_model()
    num_param = count_parameters(model)
    print(f'number of parameters: {num_param}')
    img = torch.rand((4,128,120,160))
    out = model(img)
    
