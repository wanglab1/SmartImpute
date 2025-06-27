# tests/test_smart.py

import pytest
import torch
import numpy as np

from smartimpute.smart import train_smartimpute
from smartimpute.generator import Generator
from smartimpute.discriminator import D1_WGAN_GP

@pytest.fixture
def toy_data():
    # 16 cells × 10 genes
    X = torch.rand(16, 10)
    M = (torch.rand(16, 10) > 0.5).float()
    F = torch.zeros_like(M)  # no forced zeros for real data
    return X, M, F

def test_generator_and_discriminator_can_forward(toy_data):
    X, M, _ = toy_data
    G = Generator(dim=10)
    out = G(X * M + (1 - M) * torch.rand_like(X), M)
    assert out.shape == X.shape

    D1 = D1_WGAN_GP(dim=10)
    h = M * torch.bernoulli(torch.full_like(M, 0.9))
    logits = D1(out.detach(), h)
    assert logits.shape == X.shape

def test_train_smartimpute_basic_run(toy_data):
    X, M, F = toy_data
    # run only 5 epochs and small batch for speed
    G_model, history, imputed = train_smartimpute(
        X, M, F_mask=F,
        recon_loss="mse",
        epochs=5,
        batch_size=8,
    )
    # check return types
    assert hasattr(G_model, "parameters")
    assert isinstance(history, list)
    assert isinstance(imputed, np.ndarray)
    assert imputed.shape == X.shape

    # observed entries must match input exactly
    obs = M.bool().cpu().numpy()
    np.testing.assert_allclose(imputed[obs], X.cpu().numpy()[obs], atol=1e-6)

def test_impute_fills_missing(toy_data):
    X, M, F = toy_data
    # force at least some zeros
    M[0, 0] = 0
    G_model, history, imputed = train_smartimpute(
        X, M, F_mask=F,
        recon_loss="poisson",
        epochs=5,
        batch_size=8,
    )
    missing = (~M.bool()).cpu().numpy()
    # ensure at least one missing slot got a non-zero imputation
    assert (imputed[missing] > 0).any()
