import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

# Helper functions for beta schedules
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

def cosine_beta_schedule(timesteps):
    return torch.tensor([math.cos(i / timesteps * math.pi / 2) for i in range(timesteps)])

def sigmoid_beta_schedule(timesteps):
    return torch.tensor([1 / (1 + math.exp(-10 * (i / timesteps - 0.5))) for i in range(timesteps)])

class Diffusion:
    def __init__(self, timesteps=500, beta_schedule='linear'):
        self.timesteps = timesteps
        self.betas = self._get_beta_schedule(beta_schedule)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        self._compute_diffusion_parameters()
        self._compute_posterior_parameters()

    def _get_beta_schedule(self, beta_schedule):
        if beta_schedule == 'linear':
            return linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            return cosine_beta_schedule(self.timesteps)
        elif beta_schedule == 'sigmoid':
            return sigmoid_beta_schedule(self.timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

    def _compute_diffusion_parameters(self):
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _compute_posterior_parameters(self):
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

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, cond_img, w, clip_denoised=True):
        device = next(model.parameters()).device
        batch_size = x_t.shape[0]
        
        pred_noise_cond = model(x_t, t, cond_img, torch.ones(batch_size, dtype=torch.int, device=device))
        pred_noise_uncond = model(x_t, t, cond_img, torch.zeros(batch_size, dtype=torch.int, device=device))
        pred_noise = (1 + w) * pred_noise_cond - w * pred_noise_uncond
        
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond_img, w, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond_img, w, clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_img, w=2, clip_denoised=True):
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond_img, w, clip_denoised)
            imgs.append(img.cpu().numpy())

        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, cond_img, batch_size=8, channels=3, w=2, clip_denoised=True):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), cond_img, w, clip_denoised)

    def train_losses(self, model, x_start, t, cond_img, mask_c):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, cond_img, mask_c)
        return F.mse_loss(noise, predicted_noise)