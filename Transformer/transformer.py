import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        K = self.key(x)     # (B, T, C)
        Q = self.query(x)   # (B, T, C)
        V = self.value(x)   # (B, T, C)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)  # (B, T, T)

        # Apply causal mask
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Weighted sum
        out = weights @ V  # (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size)
        self.ffwd = FeedForward(embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention + residual
        x = x + self.attn(self.ln1(x))

        # Feed-forward + residual
        x = x + self.ffwd(self.ln2(x))
        return x
if __name__ == "__main__":
    x = torch.randn(2, 8, 32)
    block = TransformerBlock(embed_dim=32, num_heads=2, block_size=8)
    y = block(x)
    print(y.shape)  # Expected: torch.Size([2, 8, 32])