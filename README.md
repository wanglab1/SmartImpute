# SmartImpute

**SmartImpute** A Targeted Imputation Framework for Single-cell Transcriptome Data


## Features

- **Imputation:** Using target gene panel for imputation. 
- **Zero Identification:** Identifies true biological zeros using probabilistic outputs.
- **Multi-Task Learning:** Combines imputation and zero classification into a unified framework.
- **Flexible Utilities:** Includes normalization, splitting, and evaluation functions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/SmartImpute.git
    cd SmartImpute
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here’s an example:

```python
from smartimpute import Generator, Discriminator, WGAIN, normalize_data, split_data

# Load and preprocess data
data = load_your_data()
data = normalize_data(data)
train_data, test_data = split_data(data)

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
    gamma=5,
    lambda_gp=10,
    epochs=100,
)

# Train and impute
wgain.train()
imputed_data, zero_probs = wgain.impute(test_data)
