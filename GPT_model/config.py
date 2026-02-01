import torch

# Model hyperparameters
batch_size = 32
block_size = 64
embed_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1

# Training
learning_rate = 3e-4
max_iters = 3000
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
