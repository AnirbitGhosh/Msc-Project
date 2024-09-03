'''
The code is modified from
https://github.com/tatakai1/classifier_free_ddim,
https://github.com/TeaPearce/Conditional_Diffusion_MNIST,

Diffusion model is based on "CLASSIFIER-FREE DIFFUSION GUIDANCE"
https://arxiv.org/abs/2207.12598,
'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

# Helper functions for beta schedules
# def linear_noise_schedule(timesteps):
#     return torch.linspace(0.0001, 0.02, timesteps)

# def cosine_noise_schedule(timesteps):
#     return torch.tensor([math.cos(i / timesteps * math.pi / 2) for i in range(timesteps)])

# def sigmoid_noise_schedule(timesteps):
#     return torch.tensor([1 / (1 + math.exp(-10 * (i / timesteps - 0.5))) for i in range(timesteps)])

# beta schedule
def linear_noise_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def sigmoid_noise_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas)/(betas.max()-betas.min())*(0.02-betas.min())/10
    return betas

def cosine_noise_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion:
    def __init__(
        self,
        timesteps=1000,
        noise_schedule='linear',
    ):
        self.timesteps = timesteps
        
        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        elif noise_schedule == 'cosine':
            betas = cosine_noise_schedule(timesteps)
        elif noise_schedule == 'sigmoid':
            betas = sigmoid_noise_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule {noise_schedule}')
        
        self.betas = betas
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion : q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # mean and variance of q(x_t | x_0)
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # mean and variance of diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: reverse of q_sample
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t) 
    def p_mean_variance(self, model, x_t, t, cond_img, w, clip_denoised=True):
        device = next(model.parameters()).device
        batch_size = x_t.shape[0]
        
        # noise prediction from model
        pred_noise_cond = model(x_t, t, cond_img, torch.ones(batch_size).int().to(device))
        pred_noise_uncond = model(x_t, t, cond_img, torch.zeros(batch_size).int().to(device))
        pred_noise = (1 + w) * pred_noise_cond - w * pred_noise_uncond
        
        # get predicted x_0
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    # denoise step: sample x_{t-1} from x_t and pred noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond_img, w, clip_denoised=True):
        # pred mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond_img, w, clip_denoised=clip_denoised)
        
        noise = torch.randn_like(x_t)
        # no noise when t = 0 
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise : reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_img, w=2, clip_denoised=True):
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), cond_img, w, clip_denoised)
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad
    def sample(self, model, image_size, cond_img, batch_size=8, channels=3, w=2, clip_denoised=True):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), cond_img, w, clip_denoised)
    
    # compute train losses
    def train_losses(self, model, x_start, t, cond_img, mask_c):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, cond_img, mask_c)
        loss = F.mse_loss(noise, predicted_noise)
        return loss 