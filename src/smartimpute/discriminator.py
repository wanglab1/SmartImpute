import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_real_fake = nn.Linear(hidden_dim, 1)
        self.fc2_zero = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        real_fake_score = self.fc2_real_fake(x)
        zero_probs = torch.sigmoid(self.fc2_zero(x))
        return real_fake_score, zero_probs
