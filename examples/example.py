from smartimpute import Generator, Discriminator, WGAIN, normalize_data, split_data, data_transformer
import pandas as pd

# Load raw data
raw_data = pd.read_csv("../data/sample_data.csv")
gene_names = raw_data.columns.tolist()
data = raw_data.values

# Define target genes
target_genes = ["GeneA", "GeneB", "GeneC", "GeneX"]  # Assume "GeneX" is not in the dataset

# Transform data: target genes first
data, new_gene_order, common_target_genes = data_transformer(data, gene_names, target_genes)
print("New Gene Order:", new_gene_order)
print("Common Target Genes:", common_target_genes)

# Normalize data
data = normalize_data(data)

# Split data
train_data, test_data = split_data(data)

# Parameters
q = len(common_target_genes)  # Automatically set q to the number of common target genes
gamma = 0.5  # Fraction of non-target genes to include during training

# Initialize WGAIN
generator = Generator(input_dim=data.shape[1], hidden_dim=128)
discriminator = Discriminator(input_dim=data.shape[1], hidden_dim=128)
wgain = WGAIN(
    data=train_data,
    generator=generator,
    discriminator=discriminator,
    batch_size=64,
    hint_rate=0.9,
    alpha=100,
    beta=10,
    gamma=gamma,
    lambda_gp=10,
    q=q,
    epochs=10,  # Set a small number of epochs for quick testing
)

# Train and impute
wgain.train()
imputed_data, zero_probs = wgain.impute(test_data)
print("Imputed Data for Target Genes:")
print(imputed_data)
print("Biological Zero Probabilities for Target Genes:")
print(zero_probs)
