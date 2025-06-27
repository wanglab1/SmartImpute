import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBase(nn.Module):
    """
    Takes as input a concatenation of the imputed (or real) data vector x
    and the corresponding hint vector h, and outputs a raw score (logit).
    """
    def __init__(self, dim, hidden_dim=None):
        """
        Args:
            dim        : number of genes (size of x and of h)
            hidden_dim : width of the hidden layers. Defaults to dim if None.
        """
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)            # raw score / logit
        )

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """
        x:    [batch_size × dim]
        hint: [batch_size × dim]
        returns: [batch_size × 1] raw score/logit
        """
        # concatenate along feature axis
        inp = torch.cat([x, hint], dim=1)
        return self.net(inp)



class D1_WGAN_GP(DiscriminatorBase):
    """
    Discriminator D1 for WGAN-GP: uses raw scores directly in 
    Wasserstein loss and gradient penalty.
    """

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        # return raw real-valued score, shape [batch_size]
        return super().forward(x, hint).view(-1)



class D2_BCE(DiscriminatorBase):
    """
    Discriminator D2 for zero-vs-dropout classification: 
    uses sigmoid(logit) and BCE loss.
    """

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        # returns probabilities in (0,1), shape [batch_size]
        logit = super().forward(x, hint).view(-1)
        return torch.sigmoid(logit)
