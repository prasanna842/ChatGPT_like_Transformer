import torch
from config import block_size, batch_size, device

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos, vocab_size
def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def decode(indices, itos):
    return "".join([itos[i] for i in indices])
def train_val_split(data, split_ratio=0.9):
    n = int(split_ratio * len(data))
    return data[:n], data[n:]

def get_batch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    
    x = torch.stack([
        data[i : i + block_size] for i in ix
    ])
    
    y = torch.stack([
        data[i + 1 : i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)

def get_dummy_batch(vocab_size):
    """
    Returns a dummy batch of token indices for testing GPT training.
    """
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    return x.to(device), y.to(device)

