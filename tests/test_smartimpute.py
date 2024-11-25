import numpy as np
import torch
from smartimpute import Generator, Discriminator, WGAIN
from smartimpute.utils import normalize_data, split_data, mask_missing_data, evaluate_zero_classification, mse_imputation_score, data_transformer

def test_generator_forward():
    input_dim = 10
    hidden_dim = 5
    generator = Generator(input_dim, hidden_dim)
    input_data = torch.rand((4, input_dim))  # Batch size 4
    imputed_values, zero_probs = generator(input_data)

    assert imputed_values.shape == input_data.shape
    assert zero_probs.shape == input_data.shape

def test_discriminator_forward():
    input_dim = 10
    hidden_dim = 5
    discriminator = Discriminator(input_dim, hidden_dim)
    input_data = torch.rand((4, input_dim))  # Batch size 4
    real_fake_score, zero_probs = discriminator(input_data)

    assert real_fake_score.shape == (4, 1)
    assert zero_probs.shape == input_data.shape

def test_data_transformer():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    gene_names = ["GeneA", "GeneB", "GeneC"]
    target_genes = ["GeneC", "GeneA"]

    reshaped_data, new_gene_order, common_target_genes = data_transformer(data, gene_names, target_genes)

    expected_data = np.array([[3, 1, 2], [6, 4, 5], [9, 7, 8]])
    expected_order = ["GeneC", "GeneA", "GeneB"]

    assert np.array_equal(reshaped_data, expected_data)
    assert new_gene_order == expected_order
    assert common_target_genes == ["GeneC", "GeneA"]

def test_wgain_train():
    input_dim = 10
    hidden_dim = 5
    generator = Generator(input_dim, hidden_dim)
    discriminator = Discriminator(input_dim, hidden_dim)
    
    data = np.random.rand(100, input_dim)
    data[data < 0.1] = np.nan  # Introduce missing values

    wgain = WGAIN(
        data=data,
        generator=generator,
        discriminator=discriminator,
        batch_size=16,
        hint_rate=0.9,
        alpha=100,
        beta=10,
        gamma=0.5,
        lambda_gp=10,
        q=5,
        epochs=1
    )
    wgain.train()

def test_mse_imputation_score():
    true_data = np.array([[1, 2], [3, np.nan]])
    imputed_data = np.array([[1, 2.1], [2.9, 4]])
    mask = mask_missing_data(true_data)
    mse = mse_imputation_score(true_data, imputed_data, mask)
    assert mse < 0.5
