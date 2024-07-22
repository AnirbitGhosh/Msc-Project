import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

def norm_layer(channels):
    return nn.GroupNorm(32, channels)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, t_emb, c_emb, mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v).reshape(B, -1, H, W)
        return self.proj(h) + x

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, cond_channels, dropout):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t, cond_img, mask):
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_cond = self.cond_conv(cond_img) * mask[:, None, None, None]
        h += emb_t[:, :, None, None] + emb_cond
        return self.conv2(h) + self.shortcut(x)