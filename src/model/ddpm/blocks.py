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

def group_norm_layer(channels):
    return nn.GroupNorm(32, channels)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepSeqEmbedding(nn.Sequential, TimestepBlock):
    def forward(self, x, time_emb, cond_emb, mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb, cond_emb, mask)
            else:
                x = layer(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, ch, heads=1):
        super(AttentionBlock, self).__init__()
        self.num_heads = heads
        assert ch % heads == 0
        
        self.norm = group_norm_layer(ch)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attention = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attention = attention.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attention, v).reshape(B, -1, H, W)
        return self.proj(h) + x

class ResBlock(TimestepBlock):
    def __init__(self, in_ch, out_ch, t_ch, cond_ch, dropout):
        super(ResBlock, self).__init__()
        
        self.conv_1 = nn.Sequential(
            group_norm_layer(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_ch, out_ch)
        )
        self.condition_conv = nn.Sequential(
            nn.Conv2d(cond_ch, out_ch, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.conv_2 = nn.Sequential(
            group_norm_layer(out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.skip_conn = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t, cond_img, mask):
        h = self.conv_1(x)
        emb_t = self.time_embedding(t)
        emb_cond = self.condition_conv(cond_img) * mask[:, None, None, None]
        h += emb_t[:, :, None, None] + emb_cond
        return self.conv_2(h) + self.skip_conn(x)