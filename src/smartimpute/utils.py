import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import Poisson

EPS = 1e-8

def gradient_penalty(D, real, fake, hint, gp_weight=10.0):
    """
    WGAN-GP gradient penalty.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp, hint)
    grads = grad(
        outputs=d_interp.sum(), inputs=interp,
        create_graph=True, retain_graph=True
    )[0]
    norm = grads.view(batch_size, -1).norm(2, dim=1)
    penalty = ((norm - 1) ** 2).mean()
    return gp_weight * penalty


def recon_loss_fn(x_obs, x_pred, mask, loss_type="mse"):
    """
    Reconstruction loss on observed entries only.
    - mse:   mean-squared error
    - poisson: negative log-likelihood under Poisson(x_pred)
    """
    if loss_type == "mse":
        return torch.mean(((mask * x_obs) - (mask * x_pred))**2) / (mask.mean() + EPS)
    elif loss_type == "poisson":
        pois = Poisson(rate=x_pred + EPS)
        nll  = -pois.log_prob(x_obs)
        return torch.mean(nll[mask.bool()])
    else:
        raise ValueError(f"Unknown recon loss: {loss_type}")
