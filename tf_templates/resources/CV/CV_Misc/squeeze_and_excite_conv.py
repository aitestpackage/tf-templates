class SE_Conv(nn.Module):
    """
    Squeeze and Excite Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=True, ffn_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            stride=stride,
            bias=bias,
        )

        self.ffn = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        weights = self.ffn(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights
    
test_se = SE_Conv(3, 8)
test_data = torch.rand(7, 3, 16, 16)
test_se(test_data).shape