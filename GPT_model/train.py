import torch
from model import GPT
from config import *

# Dummy vocab example (replace with real tokenizer)
vocab_size = 65

model = GPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Dummy data
def get_batch():
    x = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    return x, y

for step in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
