import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Define the weights for query, key, and value matrices
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)

        # Compute queries, keys, and values
        Q = self.W_q(x)  # Shape: (batch_size, seq_len, embedding_dim)
        K = self.W_k(x)  # Shape: (batch_size, seq_len, embedding_dim)
        V = self.W_v(x)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Compute attention scores
        # Q @ K.T: (batch_size, seq_len, hidden_dim) @ (batch_size, hidden_dim, seq_len)
        # Output shape: (batch_size, seq_len, seq_len)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        
        # Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, seq_len, seq_len)
        
        # Compute the weighted sum of values
        # attention_weights @ V: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, hidden_dim)
        # Output shape: (batch_size, seq_len, hidden_dim)
        attention_output = torch.bmm(attention_weights, V)
        
        return attention_output

# Example usage
batch_size = 1
seq_len = 3
embedding_dim = 4

# Random input tensor
x = torch.randn(batch_size, seq_len, embedding_dim)

# Instantiate and apply self-attention
self_attention = SelfAttention(embedding_dim)
output = self_attention(x)

print("Input:", x)
print("Self-attention output:", output)
