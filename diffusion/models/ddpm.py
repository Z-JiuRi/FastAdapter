"""See README.md for English documentation."""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_scheduler(timesteps, **kwargs):
    """See README.md for English documentation."""
    beta_start = kwargs.get("beta_start", 0.0001)
    beta_end = kwargs.get("beta_end", 0.02)
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_scheduler(timesteps, **kwargs):
    """See README.md for English documentation."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    s = kwargs.get("s", 0.008)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_scheduler(timesteps, **kwargs):
    """See README.md for English documentation."""
    beta_start = kwargs.get("beta_start", 0.0001)
    beta_end = kwargs.get("beta_end", 0.02)
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas

def get_beta_scheduler(timesteps, **kwargs):
    """See README.md for English documentation."""
    scheduler_type = kwargs.get("scheduler_type", "linear")
    
    if scheduler_type == "linear":
        return linear_beta_scheduler(timesteps, **kwargs)
    elif scheduler_type == "cosine":
        return cosine_beta_scheduler(timesteps, **kwargs)
    elif scheduler_type == "sigmoid":
        return sigmoid_beta_scheduler(timesteps, **kwargs)
    else:
        raise ValueError("Unknown beta scheduler: {}".format(scheduler_type))


def extract(a, t, x_shape: Tuple[int, ...]):
    """See README.md for English documentation."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_kwargs: Dict[str, Any] = {},
        prediction_type: str = "eps",
        snr_gamma: Optional[float] = None,
        tensor_balance: float = 0.25,
        tensor_ranges: list[tuple[int, int]] | None = None,
    ):
        """See README.md for English documentation."""
        super().__init__()
        
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        if not 0.0 <= tensor_balance <= 1.0:
            raise ValueError("tensor_balance must be in [0, 1]")
        self.tensor_balance = tensor_balance
        self.tensor_ranges = tensor_ranges or []

        # English documentation is provided in README.md.
        betas = get_beta_scheduler(timesteps, **beta_kwargs)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # English documentation is provided in README.md.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # English documentation is provided in README.md.
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # English documentation is provided in README.md.
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """See README.md for English documentation."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def predict_x0_from_v(self, x_t, t, v):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v
    
    def predict_eps_from_x0(self, x_t, t, x_0):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x_0) / sqrt_one_minus_alphas_cumprod_t
    
    def get_v(self, x_0, noise, t):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
    
    def q_posterior_mean_variance(self, x_t, t, x_0):
        """See README.md for English documentation."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def denoiser_predictions(self, x_t, t, cond, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        # English documentation is provided in README.md.
        if cfg_scale > 1.0 and uncond_cond is not None:
            x_in = torch.cat([x_t] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([uncond_cond, cond]) # English documentation is provided in README.md.
            
            structure_in = {
                key: torch.cat([value, value], dim=0) for key, value in structure.items()
            } if structure is not None else None
            
            # English documentation is provided in README.md.
            model_output = self.denoiser(x=x_in, t=t_in, token_context=c_in, structure=structure_in)
            
            out_uncond, out_cond = model_output.chunk(2)
            denoiser_output = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            # English documentation is provided in README.md.
            denoiser_output = self.denoiser(x=x_t, t=t, token_context=cond, structure=structure)
        
        if self.prediction_type == "eps":
            pred_noise = denoiser_output
            pred_x0 = self.predict_x0_from_eps(x_t, t, pred_noise)
        elif self.prediction_type == "v":
            v = denoiser_output
            pred_x0 = self.predict_x0_from_v(x_t, t, v)
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        elif self.prediction_type == "x":
            pred_x0 = denoiser_output
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t, t, cond, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        pred_noise, pred_x0 = self.denoiser_predictions(x_t, t, cond, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_t=x_t, t=t, x_0=pred_x0)
        return model_mean, posterior_variance, posterior_log_variance, pred_x0
    
    @torch.no_grad()
    def p_sample(self, x_t, t, cond, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        b = x_t.shape[0]
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x_t=x_t, t=t, cond=cond, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(b, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        return x
    
    @torch.no_grad()
    def ddim_sample(self, shape, cond, ddim_steps, eta, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        device = self.betas.device
        b = shape[0]
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x = torch.randn(shape, device=device)
        
        for time, time_prev in time_pairs:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise, pred_x0 = self.denoiser_predictions(x, t, cond, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
            
            if time_prev < 0:
                x = pred_x0
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev]
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            c = torch.sqrt(1 - alpha_prev - sigma ** 2)
            noise = torch.randn_like(x)
            
            x = torch.sqrt(alpha_prev) * pred_x0 + c * pred_noise + sigma * noise
            
        return x
    
    @torch.no_grad()
    def sample(self, cond, shape, use_ddim=False, ddim_steps=50, eta=0.0, structure=None, cfg_scale=1.0, uncond_cond=None):
        """See README.md for English documentation."""
        if shape is None:
            raise ValueError("You must explicitly provide 'shape' tuple to sample().")
            
        if use_ddim:
            return self.ddim_sample(shape, cond, ddim_steps, eta, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        else:
            return self.p_sample_loop(shape, cond, structure=structure, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
    
    def loss_fn(self, x_0, t, cond, noise=None, structure=None, token_mask=None, return_pred=False):
        """See README.md for English documentation."""
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        denoiser_output = self.denoiser(x=x_t, t=t, token_context=cond, structure=structure)
        
        if self.prediction_type == "eps":
            target = noise
        elif self.prediction_type == "x":
            target = x_0
        elif self.prediction_type == "v":
            target = self.get_v(x_0, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        squared_error = F.mse_loss(denoiser_output, target, reduction="none")
        if token_mask is None:
            token_mask = torch.ones_like(squared_error)
        else:
            token_mask = token_mask.to(dtype=squared_error.dtype)
        element_loss = (squared_error * token_mask).sum(dim=(1, 2)) / token_mask.sum(dim=(1, 2)).clamp_min(1)
        tensor_losses = []
        if structure is not None and "tensor_ids" in structure and structure["tensor_ids"].shape == squared_error.shape[:2]:
            tensor_ids = structure["tensor_ids"]
            tensor_values = torch.unique(tensor_ids, sorted=True)
            for tensor_id in tensor_values:
                tensor_mask = (tensor_ids == tensor_id).unsqueeze(-1).to(dtype=squared_error.dtype) * token_mask
                tensor_losses.append(
                    (squared_error * tensor_mask).sum(dim=(1, 2)) / tensor_mask.sum(dim=(1, 2)).clamp_min(1)
                )
        else:
            for start, count in self.tensor_ranges:
                error_slice = squared_error[:, start:start + count]
                mask_slice = token_mask[:, start:start + count]
                tensor_losses.append(
                    (error_slice * mask_slice).sum(dim=(1, 2)) / mask_slice.sum(dim=(1, 2)).clamp_min(1)
                )
        tensor_loss = torch.stack(tensor_losses, dim=1).mean(dim=1) if tensor_losses else element_loss
        loss_batch = (1 - self.tensor_balance) * element_loss + self.tensor_balance * tensor_loss
        
        # English documentation is provided in README.md.
        if self.snr_gamma is not None:
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            snr_clamped = torch.clamp(snr, max=self.snr_gamma)
            
            if self.prediction_type == 'eps':
                loss_weight = snr_clamped / snr
            elif self.prediction_type == 'v':
                loss_weight = snr_clamped / (snr + 1.0)
            else:
                # loss_weight = torch.ones_like(loss_batch)
                loss_weight = snr_clamped
            
            loss = (loss_batch * loss_weight).mean()
        else:
            loss = loss_batch.mean()
        
        
        # English documentation is provided in README.md.
        with torch.no_grad():
            flat_pred = (denoiser_output * token_mask).reshape(x_0.shape[0], -1)
            flat_target = (target * token_mask).reshape(x_0.shape[0], -1)
            cos_sim = F.cosine_similarity(flat_pred, flat_target, dim=1).mean()
            norm_sim = (torch.norm(flat_pred, dim=1) / (torch.norm(flat_target, dim=1) + 1e-8)).mean()
        
        loss_dict = {
            'loss': loss,
            'cos_sim': cos_sim,
            'norm_sim': norm_sim,
            'element_loss': element_loss.mean(),
            'tensor_loss': tensor_loss.mean(),
            'prediction_rms': denoiser_output.detach().float().square().mean().sqrt(),
            'prediction_abs_max': denoiser_output.detach().float().abs().max(),
            'target_rms': target.detach().float().square().mean().sqrt(),
        }
        if return_pred:
            loss_dict['pred'] = denoiser_output
            loss_dict['target'] = target
            if self.prediction_type == "eps":
                pred_x0 = self.predict_x0_from_eps(x_t, t, denoiser_output)
            elif self.prediction_type == "x":
                pred_x0 = denoiser_output
            elif self.prediction_type == "v":
                pred_x0 = self.predict_x0_from_v(x_t, t, denoiser_output)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            loss_dict['pred_x0'] = pred_x0
        
        return loss_dict
    
    def forward(self, x_0, cond, noise=None, structure=None, token_mask=None, return_pred=False):
        """See README.md for English documentation."""
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        result = self.loss_fn(
            x_0, t, cond, noise, structure=structure, token_mask=token_mask,
            return_pred=return_pred,
        )
        result["timestep_mean"] = t.float().mean() / max(self.timesteps - 1, 1)
        return result
