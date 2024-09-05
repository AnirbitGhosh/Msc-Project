'''
The code is modified from
https://github.com/tatakai1/classifier_free_ddim,

Diffusion model is based on "CLASSIFIER-FREE DIFFUSION GUIDANCE"
https://arxiv.org/abs/2207.12598,
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

from .blocks import AttentionBlock, ResBlock, TimestepSeqEmbedding, group_norm_layer


class Unet(nn.Module):
    def __init__(self, in_ch=2, cond_ch=1, model_ch=128, output_ch=2, res_block_num=2,  attn_res=(8, 16), dropout=0, 
                 channel_mult=(1, 2, 2, 2), conv_resample=True, heads=4):
        super().__init__()
        self.in_ch = in_ch
        self.cond_ch = cond_ch
        self.model_ch = model_ch
        self.outputput_ch = output_ch
        self.res_block_num = res_block_num
        self.attn_res = attn_res,
        self.dropout = dropout
        self.channel_mult = channel_mult,
        self.conv_resample = conv_resample
        self.heads = heads
        
        # time embedding
        time_emb_dim = model_ch * 4
        self.time_emb = nn.Sequential(
            nn.Linear(model_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # down blocks
        self.downsample_blocks = nn.ModuleList([
            TimestepSeqEmbedding(nn.Conv2d(in_ch, model_ch, kernel_size=3, padding=1))
        ])
        downsample_blocks_ch = [model_ch]
        ch = model_ch
        ds = 1
        for step, mult in enumerate(channel_mult):
            for _ in range(res_block_num):
                layers = [ResBlock(ch, model_ch * mult, time_emb_dim, cond_ch, dropout)]
                ch = model_ch * mult
                if ds in attn_res:
                    layers.append(AttentionBlock(ch, heads))
                self.downsample_blocks.append(TimestepSeqEmbedding(*layers))
                downsample_blocks_ch.append(ch)
            if step != len(channel_mult) - 1:
                self.downsample_blocks.append(TimestepSeqEmbedding(DownSample(ch, conv_resample)))
                downsample_blocks_ch.append(ch)
                ds *= 2
        
        # middle blocks
        self.mid_blocks = TimestepSeqEmbedding(
            ResBlock(ch, ch, time_emb_dim, cond_ch, dropout),
            AttentionBlock(ch, heads),
            ResBlock(ch, ch, time_emb_dim, cond_ch, dropout)
        )
        
        # up blocks
        self.upsample_blocks = nn.ModuleList([])
        for step, mult in enumerate(channel_mult[::-1]):
            for i in range(res_block_num + 1):
                layers = [
                    ResBlock(ch + downsample_blocks_ch.pop(), model_ch * mult, time_emb_dim, cond_ch, dropout)]
                ch = model_ch * mult
                if ds in attn_res:
                    layers.append(AttentionBlock(ch, heads))
                if step != len(channel_mult) - 1 and i == res_block_num:
                    layers.append(UpSample(ch, conv_resample))
                    ds //= 2
                self.upsample_blocks.append(TimestepSeqEmbedding(*layers))
                
        self.output = nn.Sequential(
            group_norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, output_ch, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps, cond_img, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond_img: a [N x cond_C x H x W] Tensor of conditional images.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        t_emb = self.time_emb(time_embedding(timesteps, dim=self.model_ch))
        
        # down step
        h = x
        for module in self.downsample_blocks:
            if cond_img.shape[2:] != h.shape[2:]:
                cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')
            h = module(h, t_emb, cond_img, mask)
            hs.append(h)
        # mid stage
        if cond_img.shape[2:] != h.shape[2:]:
            cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')
        h = self.mid_blocks(h, t_emb, cond_img, mask)
        
        # up stage
        for module in self.upsample_blocks:
            h_skip = hs.pop()
            
            if h.shape[2:] != h_skip.shape[2:]:
                h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')

            if cond_img.shape[2:] != h.shape[2:]:
                cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')

            cat_in = torch.cat([h, h_skip], dim=1)
            h = module(cat_in, t_emb, cond_img, mask)
        
        return self.output(h)
    
class UpSample(nn.Module):
    def __init__(self, ch, apply_conv):
        super(UpSample, self).__init__()
        self.apply_conv = apply_conv
        if apply_conv:
            self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.apply_conv else x

class DownSample(nn.Module):
    def __init__(self, ch, apply_conv):
        super(DownSample, self).__init__()
        self.out = nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2) if apply_conv else nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.out(x)
    
def time_embedding(timesteps, dim, max_period=1000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding if dim % 2 == 0 else torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)