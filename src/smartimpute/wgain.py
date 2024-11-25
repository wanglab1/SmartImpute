import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from .utils import mask_missing_data

class WGAIN:
    def __init__(self, data, generator, discriminator, batch_size, hint_rate, alpha, beta, gamma, lambda_gp, q, epochs):
        self.data = data
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_gp = lambda_gp
        self.q = q
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device).view(batch_size, 1)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        real_fake_score, _ = self.discriminator(interpolated)

        gradients = autograd.grad(
            outputs=real_fake_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(real_fake_score),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1))
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty

    def train(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        n_samples, n_genes = self.data.shape
        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            for i in range(0, n_samples, self.batch_size):
                batch = self.data[i:i + self.batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)

                mask = mask_missing_data(batch)

                gene_indices = list(range(self.q))
                non_target_indices = list(range(self.q, n_genes))
                np.random.shuffle(non_target_indices)
                gene_indices += non_target_indices[:int(self.gamma * (n_genes - self.q))]
                gene_indices = sorted(gene_indices)

                masked_data = batch.clone()
                masked_data[~mask] = 0

                # Train Discriminator
                optimizer_D.zero_grad()
                generated_data, zero_probs_g = self.generator(masked_data)
                combined_data = batch * mask + generated_data * ~mask

                real_fake_score_real, zero_probs_d_real = self.discriminator(batch[:, gene_indices])
                real_fake_score_fake, zero_probs_d_fake = self.discriminator(combined_data[:, gene_indices])

                gp = self.gradient_penalty(batch[:, gene_indices], combined_data[:, gene_indices])
                d_loss_real_fake = -torch.mean(real_fake_score_real) + torch.mean(real_fake_score_fake)
                d_loss_zero = nn.BCELoss()(zero_probs_d_fake[~mask[:, gene_indices]], (batch[:, gene_indices][~mask[:, gene_indices]] == 0).float())

                d_loss = d_loss_real_fake + self.lambda_gp * gp + self.beta * d_loss_zero
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                generated_data, zero_probs_g = self.generator(masked_data)
                combined_data = batch * mask + generated_data * ~mask

                impute_loss = torch.mean((batch[:, gene_indices] - generated_data[:, gene_indices])**2 * ~mask[:, gene_indices])
                zero_loss_g = nn.BCELoss()(zero_probs_g[~mask[:, gene_indices]], (batch[:, gene_indices][~mask[:, gene_indices]] == 0).float())

                real_fake_score_fake, _ = self.discriminator(combined_data[:, gene_indices])
                g_loss = -torch.mean(real_fake_score_fake) + self.alpha * impute_loss + self.beta * zero_loss_g
                g_loss.backward()
                optimizer_G.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()}, "
                  f"Impute Loss: {impute_loss.item()}, Generator Zero Loss: {zero_loss_g.item()}, "
                  f"Discriminator Zero Loss: {d_loss_zero.item()}")

    def impute(self, data):
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        mask = mask_missing_data(data)
        masked_data = data.clone()
        masked_data[~mask] = 0
        with torch.no_grad():
            imputed_data, zero_probs = self.generator(masked_data)
        imputed_data = data[:, :self.q] * mask[:, :self.q] + imputed_data[:, :self.q] * ~mask[:, :self.q]
        return imputed_data.cpu().numpy(), zero_probs[:, :self.q].cpu().numpy()
