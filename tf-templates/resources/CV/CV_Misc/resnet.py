class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels >= 16 and out_channels >= 16:
            self.skip = SE_Conv(in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=stride)
            self.conv1 = SE_Conv(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = SE_Conv(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=stride)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.silu(self.bn1(self.conv1(x)))
        x1 = F.silu(self.bn2(self.conv2(x1)))
        x = F.silu(self.skip(x))
        return x1 + x
    
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = nn.Sequential(
            ResNetBlock(3, 8, 1),
            ResNetBlock(8, 16, 2),
            ResNetBlock(16, 32, 2),
            ResNetBlock(32, 32, 1),
            nn.Dropout(0.1),
            ResNetBlock(32, 64, 2),
            nn.Dropout(0.1),
            ResNetBlock(64, 128, 1),
        )

        self.pool = nn.AdaptiveMaxPool2d((6, 6))
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(128 * 8 * 8, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)
    
sum(p.numel() for p in resnet_model.parameters())