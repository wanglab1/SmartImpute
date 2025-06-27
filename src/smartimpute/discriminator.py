import torch
import torch.nn as nn

class DiscriminatorBase(nn.Module):
    """
    Base class: concatenates x and hint, returns raw logit.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hint):
        # x, hint: [batch × dim]
        return self.net(torch.cat([x, hint], dim=1)).view(-1)


class D1_WGAN_GP(DiscriminatorBase):
    """
    D1: for Wasserstein GAN with gradient penalty.
    Returns a real-valued score.
    """
    def forward(self, x, hint):
        return super().forward(x, hint)


class D2_BCE(DiscriminatorBase):
    """
    D2: for binary classification (true vs forced zeros).
    Applies sigmoid to the logit.
    """
    def forward(self, x, hint):
        logit = super().forward(x, hint)
        return torch.sigmoid(logit)
