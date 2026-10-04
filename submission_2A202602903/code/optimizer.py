"""Optimizer factories and global gradient clipping."""
import torch

OPTIMIZERS = ('sgd', 'sgd_momentum', 'adam', 'adamw')

def build_optimizer(name, params, lr, weight_decay=0., momentum=.9, betas=(.9,.999), eps=1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(name)
    if name.startswith('sgd'):
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum if name == 'sgd_momentum' else 0.)
    cls = torch.optim.Adam if name == 'adam' else torch.optim.AdamW
    return cls(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

def build_scheduler(optimizer, name, total_steps, **kwargs):
    if name is None:
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, **kwargs)
    raise ValueError(name)

def clip_gradients(params, max_norm):
    return float(torch.nn.utils.clip_grad_norm_(params, float('inf') if max_norm is None else max_norm))
