import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_impute = nn.Linear(hidden_dim, input_dim)
        self.fc2_zero = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        imputed_values = torch.sigmoid(self.fc2_impute(x))
        zero_probs = torch.sigmoid(self.fc2_zero(x))
        return imputed_values, zero_probs
