# !pip install muon_optimizer
# !pip install sacrebleu

def relu2(x):
    x = F.relu(x)
    # x = F.gelu(x)
    x = torch.square(x)
    return x

class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.relu(x)
        x = torch.square(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            ReLU2(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.alpha = nn.Parameter(torch.tensor([1.0, 1.0]) * 1.0)

    def forward(self, x, mask):
        x1 = self.norm1(x)
        # x1 = x
        x1, _ = self.self_attn(x1, x1, x1, attn_mask=mask)
        x = x + x1 * self.alpha[0]
        x1 = self.norm2(x)
        x1 = self.ffn(x1)
        return x + x1 * self.alpha[1]
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            ReLU2(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.alpha = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]) * 1.0)

    def forward(self, x, memory, tgt_mask):
        x1 = self.norm1(x)
        x1, _ = self.self_attn(x1, x1, x1, attn_mask=tgt_mask)
        x1 = self.dropout1(x1)
        x = x + x1 * self.alpha[0]
        x1 = self.norm2(x)
        x1, _ = self.cross_attn(x1, memory, memory, attn_mask=None) # No need to mask encoder memory in NMT
        x1 = self.dropout2(x1)
        x = x + x1 * self.alpha[1]
        x = self.norm3(x)
        x = x + self.ffn(x) * self.alpha[2]
        return x
    
import torch.nn as nn
import torch, math
from einops import rearrange
import torch.nn.functional as F

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()

        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU()
        ))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU()
            ))
        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device):
        indices = (n-1) + torch.arange(n).unsqueeze(1) - torch.arange(n).unsqueeze(0)
        pos = torch.arange(-n + 1, n, device = device).float().unsqueeze(-1)
        for layer in self.mlp:
            pos = layer(pos)
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
    
class EncoderDecoderwAlibi(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, in_vocab_size, out_vocab_size, seq_len, num_layers):
        super().__init__()
        self.src_embedding = nn.Embedding(in_vocab_size, hidden_dim)
        self.trg_embedding = nn.Embedding(out_vocab_size, hidden_dim)

        self.encoder = nn.ModuleList([EncoderLayer(hidden_dim, 8, ffn_dim) for _ in range(num_layers[0])])
        self.decoder = nn.ModuleList([DecoderLayer(hidden_dim, 8, ffn_dim) for _ in range(num_layers[1])])
        
        self.alibi_m = [1 / (2**i) for i in range(1, 9)]
        x = torch.arange(seq_len)
        y = torch.arange(seq_len).unsqueeze(-1)
        self.alibi_val = x - y
        self.alibi_val = self.alibi_val.to(device).unsqueeze(0)
        self.alibi_val.requires_grad = False
        
        self.pe = DynamicPositionBias(hidden_dim // 4, 8, 3)
        # self.pe.require_grad = False
        
        self.causal_mask = torch.ones(1, seq_len, seq_len, requires_grad=False, device=device) * (float('-inf'))
        self.causal_mask = torch.triu(self.causal_mask, diagonal=1)
        
        self.output = nn.Linear(hidden_dim, out_vocab_size)

    def forward(self, src, trg):
        batch_size = src.shape[0]
        x = self.src_embedding(src)

        alibi_mask = self.pe(x.shape[1], device=device)
        alibi_mask = alibi_mask.repeat(batch_size, 1, 1)
        # END MASK COMPUTATION

        # x = self.encoder(x, mask=alibi_mask)
        for layer in self.encoder:
            x = layer(x, mask=alibi_mask)
        # x = F.rms_norm(x, (x.size(-1),))

        # DECODER MASK
        mask = self.causal_mask.expand(batch_size * 8, -1, -1)
        mask = mask + alibi_mask
        # END OF DECODER MASK
        
        trg = self.trg_embedding(trg)
        trg_len = trg.shape[1]
        mask = mask[:, :trg_len, :trg_len]
        for layer in self.decoder:
            trg = layer(trg, x, mask)
        return self.output(trg)
    
model = EncoderDecoderwAlibi(
    hidden_dim=32,
    ffn_dim=16,
    in_vocab_size=2048, 
    out_vocab_size=2048, 
    seq_len=32, 
    num_layers=(1, 1),
).to(device)