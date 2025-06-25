class PatchImage(nn.Module):
    def __init__(self, in_channels, width, height, p_size, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.to_patch = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=(p_size, p_size), stride=p_size, padding=0, bias=False, groups=16),
            nn.BatchNorm2d(hidden_dim),
        )

        num_w = width // p_size
        num_h = height // p_size
        self.num_patches = num_w * num_h
        grid_w, grid_h = torch.meshgrid(torch.arange(num_w) + 1, torch.arange(num_h) + 1)
        grid = grid_w + grid_h
        grid = grid.unsqueeze(-1).repeat(1, 1, hidden_dim)
        scaling = torch.arange(hidden_dim) + 1
        scaling = scaling.unsqueeze(0).unsqueeze(0) / math.sqrt(width)
        pos_embed = torch.sin(grid * scaling).unsqueeze(0).permute(0, 3, 1, 2) # unsqueeze for batch size
        self.pos_embedding = nn.Parameter(pos_embed)
        self.embed_weight = nn.Parameter(torch.tensor(0.05)) # REMEMBER TO ADD THIS
        # IF NOT THE ORIGINAL IMAGE'S SIGNAL WILL BE MASKED

    def forward(self, x):
        x = self.pre_conv(x)
        # x: [batch size, 16, width, height]
        # print(self.to_patch(x).shape)
        # print(self.pos_embedding.shape)
        x = self.to_patch(x) + self.pos_embedding * self.embed_weight
        # x: [batch size, hidden_dim, num_w, num_h]
        x = x.reshape(-1, self.hidden_dim, self.num_patches)
        # x: [batch size, hidden_dim, num_w * num_h]
        x = x.permute(0, 2, 1)
        # x: [batch size, num_w * num_h, hidden_dim]
        return F.silu(x)
    
class Vanilla_ViT(nn.Module):
    def __init__(self, width, height, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.to_patch = PatchImage(3, width, height, 8, hidden_dim)
        self.num_patches = self.to_patch.num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
            activation=F.gelu,
            dropout=0.25
        )
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pooling = nn.Conv2d(1, 1, kernel_size=(self.num_patches, 1), stride=1, padding=0)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.to_patch(x)
        # x: [batch size, num_patches, hidden_dim]
        x = self.transformer_layers(x)
        # x: [batch size, num_patches, hidden_dim]
        x = x.unsqueeze(1)
        # x: [batch size, 1, num_patches, hidden_dim]
        x = F.gelu(self.pooling(x))
        # x: [batch size, 1, 1, hidden_dim]
        x = x.squeeze()
        # x: [batch size, hidden_dim]
        return self.fc(x)