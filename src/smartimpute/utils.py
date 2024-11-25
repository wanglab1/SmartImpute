import numpy as np
import pandas as pd
import torch

def normalize_data(data):
    """
    Normalize data to the range [0, 1].
    """
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def split_data(data, test_size=0.2, random_seed=42):
    """
    Split data into training and testing sets.
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * (1 - test_size))
    return data[indices[:split_idx]], data[indices[split_idx:]]

def mask_missing_data(data):
    """
    Generate a mask for missing data (NaN values).
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    return ~torch.isnan(data)

def evaluate_zero_classification(true_data, predicted_probs, threshold=0.5):
    """
    Evaluate zero classification using precision, recall, and F1-score.
    """
    true_labels = (true_data == 0).astype(int)
    predicted_labels = (predicted_probs >= threshold).astype(int)

    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def mse_imputation_score(true_data, imputed_data, mask):
    """
    Compute the Mean Squared Error (MSE) for imputation.
    """
    true_values = true_data[~mask.numpy()]
    imputed_values = imputed_data[~mask.numpy()]
    mse = np.mean((true_values - imputed_values) ** 2)
    return mse

def data_transformer(data, gene_names, target_genes):
    """
    Reorder the data matrix such that common target genes are the first `q` genes.
    """
    if isinstance(data, pd.DataFrame):
        gene_names = data.columns.tolist()
        data = data.values

    # Find common target genes
    common_target_genes = list(set(target_genes).intersection(set(gene_names)))
    if not common_target_genes:
        raise ValueError("No common genes found between the target gene panel and the dataset.")

    # Find indices of common target genes and non-target genes
    gene_name_to_index = {gene: idx for idx, gene in enumerate(gene_names)}
    target_indices = [gene_name_to_index[gene] for gene in common_target_genes]
    non_target_indices = [idx for idx in range(len(gene_names)) if idx not in target_indices]

    # Combine indices: common target genes first, then non-target genes
    new_order = target_indices + non_target_indices
    reshaped_data = data[:, new_order]

    # Reorder the gene names accordingly
    new_gene_order = [gene_names[idx] for idx in new_order]

    return reshaped_data, new_gene_order, common_target_genes
