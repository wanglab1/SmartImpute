import torch
from torch import nn

class Generator(nn.Module):
    """
    Generator network: takes masked data + mask hint and predicts imputed values.
    """
    def __init__(self, dim, use_zinb_pi: bool = False):
        """
        dim: number of genes/features
        use_zinb_pi: whether to output a dropout probability head for ZINB
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.use_zinb_pi = use_zinb_pi
        if self.use_zinb_pi:
            self.pi_head = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )

    def forward(self, x, m):
        inp = torch.cat([x, m], dim=1)
        mu = self.net(inp)
        if self.use_zinb_pi:
            pi = self.pi_head(inp)
            return mu, pi
        else:
            return mu
