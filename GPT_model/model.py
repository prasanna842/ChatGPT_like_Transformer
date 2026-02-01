import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape

        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, dropout=0.1):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
