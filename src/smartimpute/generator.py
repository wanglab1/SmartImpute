import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Base GAIN generator: imputes missing entries.
    If use_zinb_pi=True, returns (mu, pi) for ZINB.
    """
    def __init__(self, dim, hidden_dim=None, use_zinb_pi=False):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.use_zinb_pi = use_zinb_pi

        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        if use_zinb_pi:
            self.pi_head = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim),
                nn.Sigmoid()
            )

    def forward(self, x, mask):
        """
        x:    [batch × dim] with masked entries replaced by noise
        mask: [batch × dim] binary mask (1=observed, 0=missing)
        returns:
           mu_pred [batch × dim]
           optionally pi_pred [batch × dim] if use_zinb_pi
        """
        inp = torch.cat([x, mask], dim=1)
        mu = self.net(inp)
        if self.use_zinb_pi:
            pi = self.pi_head(inp)
            return mu, pi
        return mu

