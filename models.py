#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:02:59 2021

@author: weiyunjiang
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class VGG_16(nn.Module):
    """ Naive VGG-16
    
        
    """
    def __init__(self, output_size=(320, 240)):
        super(VGG_16, self).__init__()
        self.output_size = output_size
        self.vgg = vgg16_bn(pretrained=True)
        self.vgg.classifier._modules['6'] = nn.Linear(4096, output_size[0]*output_size[1])
        self.transform = torch.nn.functional.interpolate

        
    def forward(self, image):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.vgg(image)
        out = F.relu(out)
        out += 1e-3
        out = out.view(-1, *self.output_size)
        return out

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        n_patches = int((160 // patch_size) * (120 // patch_size))
        self.pos_enc = nn.Parameter(torch.rand((1, embedding_dim, n_patches)))

    def forward(self, x):
        # x (N, num_decoded_ch, 240, 320)
        embeddings = self.embedding_convPxP(x)  # (N, embedding_dim, 240/patch_size, 320/patch_size)
        embeddings = embeddings.flatten(2) # (N, embedding_dim, n_patches) 
        embeddings += self.pos_enc # (N, embedding_dim, n_patches)
        # change to (n_patches, N, embedding_dim) format required by transformer
        embeddings = embeddings.permute(2, 0, 1)  # (n_patches, N, embedding_dim)
        x = self.transformer_encoder(embeddings)  # (n_patches, N, embedding_dim)
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        N, C, H, W = x.size()
        _, Cout, Ck = K.size()
        assert C == Ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        out = x.view(N, C, H * W).permute(0, 2, 1) @ K.permute(0, 2, 1)  # (N, H*W, Cout)
        return out.permute(0, 2, 1).view(N, Cout, H, W)

class mViT(nn.Module):
    def __init__(self, num_decoded_ch, n_query_channels=128, patch_size=16, dim_out=100,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(num_decoded_ch, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(num_decoded_ch, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.MLP = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # x (N, num_decoded_ch, 240, 320)
        tgt = self.patch_transformer(x.clone())  # (n_patches, N, num_decoded_ch)

        x = self.conv3x3(x) # (N, num_decoded_ch, 240, 320)

        head, queries = tgt[0, :], tgt[1:self.n_query_channels + 1, :]
        # regression_head (N, num_decoded_ch)
        # queries (n_query_channels, N, num_decoded_ch)
        # discard rest of transformer output
        
        # (n_query_channels, N, num_decoded_ch) --> (N, n_query_channels, num_decoded_ch)
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # (N, n_query_channels, 240, 320)

        bin_widths = self.MLP(head)  # (N, n_bins)
        if self.norm == 'linear':
            bin_widths = torch.relu(bin_widths)
            eps = 0.1
            bin_widths = bin_widths + eps
        elif self.norm == 'softmax':
            return torch.softmax(bin_widths, dim=1), range_attention_maps
        else:
            bin_widths = torch.sigmoid(bin_widths)
        bin_widths_normed = bin_widths / bin_widths.sum(dim=1, keepdim=True)
        return bin_widths_normed, range_attention_maps

class UpSample(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(UpSample, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_ch, output_ch, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.LeakyReLU(),
                                  )

    def forward(self, x, concat_block):
        up_x = F.interpolate(x, size=[concat_block.size(2), concat_block.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_block], dim=1)
        return self._net(f)


class Decoder(nn.Module):
    def __init__(self, ch=2048, num_decoded_ch=128):
        super(Decoder, self).__init__()

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=1, padding=1)

        self.up1 = UpSample(input_ch=int(ch / 1 + 112 + 64), output_ch=int(ch / 2))
        self.up2 = UpSample(input_ch=int(ch / 2 + 40 + 24), output_ch=int(ch / 4))
        self.up3 = UpSample(input_ch=int(ch / 4 + 24 + 16), output_ch=int(ch / 8))
        self.up4 = UpSample(input_ch=int(ch / 8 + 16 + 8), output_ch=int(ch / 16))

        self.conv3 = nn.Conv2d(int(ch / 16), num_decoded_ch, kernel_size=3, padding=1)
        
    def forward(self, features):
        #x (2,3,480,640)
        #x_block0 (2,24,240,320)
        #x_block1 (2,40,120,160)
        #x_block2 (2,64,60,80)
        #x_block3 (2,176,30,40)
        #x_block4 (2,2048,15,20)
        x_block0, x_block1, x_block2, x_block3, x_block4 = \
            features[4], features[5], features[6], features[8], features[11]
        
        #(2,2048,17,22)
        x_d0 = self.conv2(x_block4)
        #(2,1024,30,40)
        x_d1 = self.up1(x_d0, x_block3)
        #(2,512,60,80)
        x_d2 = self.up2(x_d1, x_block2)
        #(2,256,120,160)
        x_d3 = self.up3(x_d2, x_block1)
        #(2,128,240,320)
        x_d4 = self.up4(x_d3, x_block0)
        
        #(2,128,240,320)
        out = self.conv3(x_d4)
        return out



class Encoder(nn.Module):
    # extract the skip-layer outputs
    def __init__(self, encoder_model):
        super(Encoder, self).__init__()
        self.encoder_model = encoder_model

    def forward(self, x):
        features = [x]
        for key, value in self.encoder_model._modules.items():
            if (key == 'blocks'):
                for keyb, valueb in value._modules.items():
                    features.append(valueb(features[-1]))
            else:
                features.append(value(features[-1]))
        return features

class UnetAdaptiveBins(nn.Module):
    def __init__(self, encoder_model, num_decoded_ch=128, n_bins=100, 
                 min_val=0.1, max_val=10, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(encoder_model)
        self.decoder = Decoder(num_decoded_ch=num_decoded_ch)
        self.adaptive_bins_layer = mViT(num_decoded_ch=num_decoded_ch, n_query_channels=128, 
                                        patch_size=16, dim_out=n_bins, embedding_dim=num_decoded_ch,
                                        norm=norm)

        
        self.conv_out = nn.Sequential(nn.Conv2d(num_decoded_ch, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1),
                                      )

    def forward(self, x, **kwargs):
        decoded_features = self.decoder(self.encoder(x), **kwargs) # (N, num_decoded_ch, 240, 320)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(decoded_features)
        # bin_widths_normed (2, n_bins), range_attention_maps (N, num_decoded_ch, 240, 320)
        range_bins_maps = self.conv_out(range_attention_maps) # (N, n_bins, 240, 320)

        # Post process
        # N, n_bins, H, W = range_bins_maps.shape
        # hist = torch.sum(range_bins_maps.view(N, n_bins, H * W), dim=2) / (H * W)  # not used for training
        
        # map the bin intervals
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # (N, n_bins)
        # add the min depth value to the front
        bin_widths = nn.functional.pad(bin_widths, (1, 0), value=self.min_val) # (N, n_bins + 1)
        bin_edges = torch.cumsum(bin_widths, dim=1) # (N, n_bins + 1)
        # compute the width by taking the average of two adjacent bin widths
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) # (N, n_bins)
        N, n_bins = centers.size()
        centers = centers.view(N, n_bins, 1, 1) # (N, n_bins, 1, 1)

        pred = torch.sum(range_bins_maps * centers, dim=1, keepdim=True) # (N, 1, 240, 320)

        return centers.view(N, n_bins, 1), pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build_encoder(cls, n_bins, **kwargs):
        encoder_name = 'tf_efficientnet_b5_ap'

        print('Loading pre-trained encoder model ()...'.format(encoder_name), end='')
        encoder_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', encoder_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        encoder_model.global_pool = nn.Identity()
        encoder_model.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(encoder_model, n_bins=n_bins, **kwargs)
        print('Done.')
        return m
    
if __name__ == '__main__':
    model = UnetAdaptiveBins.build_encoder(n_bins=80)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
