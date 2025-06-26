class PatchConvLayer(nn.Module):
    # Basically the same at the PatchImage class above
    # Except no positional Embedding this time(?)
    def __init__(self, in_channels, width, height, p_size, out_channels):
        super().__init__()  
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.to_patch = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=(p_size, p_size), stride=p_size, padding=0, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.pre_conv(x)
        return self.to_patch(x)
    
class ConvMixLayer(nn.Module):
    def __init__(self, n_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            n_channels, 
            n_channels, 
            kernel_size=(kernel_size, kernel_size), 
            stride=1, 
            padding='same', 
            groups=n_channels
        )
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.pointwise = nn.Conv2d(n_channels, n_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        x1 = F.silu(self.depthwise(x))
        x1 = self.bn1(x1)
        x = x + x1
        x = self.pointwise(x)
        x = F.silu(x)
        return self.bn2(x)
    
class ConvMixer(nn.Module):
    def __init__(self, n_channels, depth, patch_size, n_classes, kernel_size=3):
        super().__init__()
        self.to_patch = PatchConvLayer(3, 128, 128, patch_size, n_channels)

        self.convmix_layers = nn.ModuleList(
            [ConvMixLayer(n_channels, kernel_size) for _ in range(depth)]
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(n_channels * 6 * 6, 128),
            nn.GELU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.to_patch(x)
        for layer in self.convmix_layers:
            x = layer(x)
        return self.pooling(x)

convmix_model = ConvMixer(128, 2, 16, 6, 3).to(device)
convmix_criteria = nn.CrossEntropyLoss(label_smoothing=0.1)
# convmix_optimiser = optim.SGD(convmix_model.parameters(), lr=1e-3, momentum=0.9)
convmix_optimiser = optim.AdamW(convmix_model.parameters(), lr=3e-4, weight_decay=0.01)
convmix_scheduler = optim.lr_scheduler.ExponentialLR(convmix_optimiser, gamma=0.9)