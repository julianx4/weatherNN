import math
import torch
import torch.nn as nn
import torch.optim as optim

def generate_square_subsequent_mask(sz):
    mask = torch.tril(torch.ones(sz, sz))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                    
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(k, heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, value, key, query, mask=None):
        if mask is not None:
            att = self.attention(query, key, value, attn_mask=mask)[0]
        else:
            att = self.attention(query, key, value)[0]
        x = self.norm1(att + value)
        ff = self.ff(x)
        return self.norm2(ff + x)

class Predictor(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_features, max_seq_len=80):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.seq_length = seq_length

        self.pos_encoder = PositionalEncoder(k, max_seq_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(k, heads) for _ in range(depth)
        ])

        self.linear = nn.Linear(k, num_features)

    def forward(self, x):
        # x has shape (batch_size, seq_length, num_features)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, num_features)
        x = self.pos_encoder(x)  # Add positional encodings

        mask = generate_square_subsequent_mask(x.size(0)).to(x.device)

        # Apply the transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, x, x, mask)

        x = self.linear(x.permute(1, 0, 2).reshape(-1, self.k))

        return x.reshape(-1, self.seq_length, self.num_features)