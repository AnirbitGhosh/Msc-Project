import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

from blocks import AttentionBlock, ResidualBlock, TimestepEmbedSequential, norm_layer



class Unet(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, model_channels=128, out_channels=2, num_res_blocks=2, 
                 attention_resolutions=[], dropout=0, channel_mult=(1, 2, 2), conv_resample=True, num_heads=2):
        super(Unet, self).__init__()
        
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.down_blocks, down_block_channels = self._build_down_blocks()
        self.middle_blocks = self._build_middle_blocks(down_block_channels[-1], time_emb_dim)
        self.up_blocks = self._build_up_blocks(down_block_channels, time_emb_dim)
        self.out = self._build_output_layer(down_block_channels[0])

    def _build_down_blocks(self):
        down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1))
        ])
        
        down_block_channels = [self.model_channels]
        ch = self.model_channels
        ds = 1
        
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [ResidualBlock(ch, self.model_channels * mult, self.model_channels * 4, self.cond_channels, self.dropout)]
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, self.num_heads))
                down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_channels.append(ch)
            if level != len(self.channel_mult) - 1:
                down_blocks.append(TimestepEmbedSequential(DownSample(ch, self.conv_resample)))
                down_block_channels.append(ch)
                ds *= 2
                
        return down_blocks, down_block_channels

    def _build_middle_blocks(self, ch, time_emb_dim):
        return TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_emb_dim, self.cond_channels, self.dropout),
            AttentionBlock(ch, self.num_heads),
            ResidualBlock(ch, ch, time_emb_dim, self.cond_channels, self.dropout)
        )

    def _build_up_blocks(self, down_block_channels, time_emb_dim):
        up_blocks = nn.ModuleList([])
        ds = 2 ** (len(self.channel_mult) - 1)
        
        for level, mult in enumerate(self.channel_mult[::-1]):
            for i in range(self.num_res_blocks + 1):
                layers = [
                    ResidualBlock(down_block_channels.pop() + (self.model_channels * mult if i == 0 else 0), 
                                  self.model_channels * mult, time_emb_dim, self.cond_channels, self.dropout)]
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(self.model_channels * mult, self.num_heads))
                if level != len(self.channel_mult) - 1 and i == self.num_res_blocks:
                    layers.append(UpSample(self.model_channels * mult, self.conv_resample))
                    ds //= 2
                up_blocks.append(TimestepEmbedSequential(*layers))
                
        return up_blocks

    def _build_output_layer(self, ch):
        return nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps, cond_img, mask):
        hs = []
        t_emb = self.time_emb(timestep_embedding(timesteps, self.model_channels))
        
        h = x
        for module in self.down_blocks:
            if cond_img.shape[2:] != h.shape[2:]:
                cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')
            h = module(h, t_emb, cond_img, mask)
            hs.append(h)
            
        if cond_img.shape[2:] != h.shape[2:]:
            cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')
        h = self.middle_blocks(h, t_emb, cond_img, mask)
        
        for module in self.up_blocks:
            h_skip = hs.pop()
            if h.shape[2:] != h_skip.shape[2:]:
                h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
            if cond_img.shape[2:] != h.shape[2:]:
                cond_img = F.interpolate(cond_img, size=h.shape[2:], mode='nearest')
            h = module(torch.cat([h, h_skip], dim=1), t_emb, cond_img, mask)
        
        return self.out(h)
    
class UpSample(nn.Module):
    def __init__(self, channels, use_conv):
        super(UpSample, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x

class DownSample(nn.Module):
    def __init__(self, channels, use_conv):
        super(DownSample, self).__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2) if use_conv else nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.op(x)
    
def timestep_embedding(timesteps, dim, max_period=1000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding if dim % 2 == 0 else torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)