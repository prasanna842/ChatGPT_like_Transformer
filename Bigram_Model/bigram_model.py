import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== Load dataset =======
# Replace 'input.txt' with your own text file if you want
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# build the vocabulary list of all characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# maps from char -> index and index -> char
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

# encode entire dataset to integer indices
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# batch generator for bigram: (x, y)
def get_batch(split, batch_size=32):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - 1, (batch_size,))
    x = data[ix]      # current char index
    y = data[ix + 1]  # next char index
    return x, y

# ====== Bigram Model =======
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each row in embedding is a log-prob vector for next char
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        # idx shape: (batch_size)
        logits = self.token_embedding_table(idx)  # (batch_size, vocab_size)
        return logits

model = BigramModel(vocab_size)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for step in range(2000):
    xb, yb = get_batch('train')
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# ====== Text Generation =======
def generate(model, length=500):
    model.eval()

    idx = torch.randint(vocab_size, (1,), dtype=torch.long)
    out = itos[int(idx)]

    for _ in range(length):
        logits = model(idx)                 # (1, vocab)
        probs = F.softmax(logits, dim=-1)   # (1, vocab)

        idx = torch.multinomial(probs, num_samples=1).squeeze()
        idx = idx.unsqueeze(0)              # keep shape (1,)

        out += itos[int(idx)]

    return out

print("\n=== Generated Text ===")
print(generate(model, length=500))
