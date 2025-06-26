class TransformerModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim, num_heads, num_layers, seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)

        position_enc = np.array(
            [
                [pos / np.power(10000, 2*i/embed_dim) for i in range(embed_dim)] 
                if pos != 0 else np.zeros(embed_dim) for pos in range(seq_len)
            ]
        )
        
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)
        self.positional_embedding = nn.Parameter(position_enc)
        # self.positional_embedding = position_enc

        one_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            activation=F.gelu,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer_layers = nn.TransformerEncoder(one_layer, num_layers=num_layers)

        self.conv_pool = nn.Conv2d(1, 1, kernel_size=(seq_len, 1), padding=0, stride=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # x: [batch size, seq len, embed dim]
        x = x + self.positional_embedding
        # x: [batch size, seq len, embed dim]
        x = F.gelu(self.proj(x))
        # x: [batch size, seq len, hidden dim]
        x = self.transformer_layers(x)
        # x: [batch size, seq len, hidden dim]
        x = x.unsqueeze(1)
        # x: [batch size, 1, seq len, hidden dim]
        x = self.conv_pool(x)
        # x: [batch size, 1, 1, hidden dim]
        x = x.squeeze()
        # x: [batch size, hidden dim]
        return self.fc(x)