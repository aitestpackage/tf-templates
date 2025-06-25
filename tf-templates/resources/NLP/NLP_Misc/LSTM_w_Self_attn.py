class LSTM_with_Attention(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_classes):
        super(LSTM_with_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_1 = nn.LSTM(input_size=embedding_dim, 
                              hidden_size=hidden_dim,
                              num_layers=2,
                              bidirectional=False,
                              dropout=0,
                              batch_first=True)
        self.attention_1 = SelfAttention(hidden_dim=hidden_dim)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, 
                              hidden_size=hidden_dim,
                              num_layers=2,
                              bidirectional=False,
                              dropout=0,
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
#         print("x:", x.shape)
        embedded = self.embedding(x)
#         print(embedded.shape)
        output, (ht, ct) = self.lstm_1(embedded)
#         print("output:", output.shape)
#         print("ht[-1]:", ht[-1].shape)
#         print("ct:", ct.shape)
#         print("self.fc(ht[-1]):", self.fc(ht[-1]).shape)
#         print("output, ht, ct shapes:", output.shape, ht.shape, ct.shape)
        attention_output = self.attention_1(output)
#         print("attention_output:", attention_output.shape)
        output2, (ht2, ct2) = self.lstm_2(attention_output)
#         print(ht2[-1].shape)
        return self.fc(ht2[-1])
