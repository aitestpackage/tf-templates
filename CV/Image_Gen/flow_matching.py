class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=False):
        super().__init__()
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1)
            self.skip = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, stride=stride, kernel_size=4, padding=1)
            # self.conv2 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear'),
            #     nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1),
            # )
            self.skip = nn.ConvTranspose2d(in_channels, out_channels, stride=stride, kernel_size=4, padding=1)
            # self.skip = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear'),
            #     nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1),
            # )
        g = min(32, out_channels)
        self.bn1 = nn.GroupNorm(g, out_channels)
        self.bn2 = nn.GroupNorm(g, out_channels)

        # self.alpha = nn.Parameter(torch.tensor([0.0]))

        self.gamma_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, out_channels),
            nn.Tanh(),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, out_channels),
            nn.Tanh(),
        )
    def forward(self, x, t_embed):
        gamma = self.gamma_mlp(t_embed).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_mlp(t_embed).unsqueeze(-1).unsqueeze(-1)
        # x = x * (1 + gamma) + beta
        x1 = F.silu(self.bn1(self.conv1(x)))
        # x1 = x1 * (1 + gamma) + beta
        x1 = F.silu(self.bn2(self.conv2(x1)))
        # x1 = x1 * (1 + gamma) + beta
        x = self.skip(x) + x1
        x = x * (1 + gamma) + beta
        return x
    
class ViT(nn.Module):
    # Patch size = 1
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self
        self.to_patch = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1,
            padding=0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            dim_feedforward=128,
            nhead=2,
            activation=nn.GELU(),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        x = self.to_patch(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = ResNetBlock(3, 16, 1)
        self.down1 = ResNetBlock(16, 32, 2)
        self.down2 = ResNetBlock(32, 64, 2)
        self.down3 = ResNetBlock(64, 128, 2)
        self.d3_vit = ViT(64, 64)
        self.bottleneck = ResNetBlock(128, 128, 1)
        self.b_vit = ViT(128, 128)
        self.up2 = ResNetBlock(256, 128, 2, upsample=True)
        self.u2_vit = ViT(128, 128)
        self.up2_c = ResNetBlock(128, 64, 1)
        self.up3 = ResNetBlock(128, 64, 2, upsample=True)
        self.up3_c = ResNetBlock(64, 32, 1)
        self.up4 = ResNetBlock(64, 32, 2, upsample=True)
        self.up4_c = ResNetBlock(32, 32, 1)
        self.out_conv = ResNetBlock(32, 3, 1)
    def forward(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
            t = t.unsqueeze(-1)
        time_embed = torch.exp(torch.arange(128) / 16).unsqueeze(0).expand(x.shape[0], -1)
        time_embed = time_embed.to(device)
        time_embed = (t.expand(-1, 128)) / time_embed
        time_embed = torch.sin(time_embed)
        x0 = self.pre_conv(x, time_embed)
        x1 = self.down1(x0, time_embed)
        x2 = self.down2(x1, time_embed)
        x2 = self.d3_vit(x2) + x2
        x3 = self.down3(x2, time_embed)
        x4 = self.bottleneck(x3, time_embed) + self.b_vit(x3)
        y = torch.cat([x3, x4], dim=1)
        y = self.up2(y, time_embed)
        y = self.u2_vit(y) + y
        y = self.up2_c(y, time_embed)
        y = torch.cat([y, x2], dim=1)
        y = self.up3(y, time_embed)
        y = self.up3_c(y, time_embed)
        y = torch.cat([y, x1], dim=1)
        y = self.up4(y, time_embed)
        y = self.up4_c(y, time_embed)
        y = self.out_conv(y, time_embed)
        return y

def train():
    model.train()
    total_loss = 0
    cnt = 0
    for img, _ in (pbar := tqdm(train_loader)):
        img = img.to(device)
        img = img * 2 - 1
        x0 = torch.randn(img.shape).to(device)
        t = torch.rand(img.shape[0], 1).to(device)
        t2 = t.detach().clone().unsqueeze(-1).unsqueeze(-1)
        x_t = t2 * img + (1 - t2) * x0
        v_target = img - x0
        v_pred = model(t, x_t)
        loss = criteria(v_pred.reshape(-1), v_target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cnt += 1
        pbar.set_description(f"Training Loss: {total_loss / cnt :6f}")
    scheduler.step()

def test():
    model.eval()
    total_loss = 0
    cnt = 0
    with torch.no_grad():
        for img, _ in (pbar := tqdm(test_loader)):
            img = img.to(device)
            img = img * 2 - 1
            x0 = torch.randn(img.shape).to(device)
            t = torch.rand(img.shape[0], 1).to(device)
            t2 = t.detach().clone().unsqueeze(-1).unsqueeze(-1)
            # print(f"t.shape: {t.shape}")
            # print(f"x0.shape: {x0.shape}")
            # print(f"img.shape: {img.shape}")
            x_t = t2 * img + (1 - t2) * x0
            v_target = img - x0
            v_pred = model(t, x_t)
            loss = criteria(v_pred.reshape(-1), v_target.reshape(-1))
            total_loss += loss.item()
            cnt += 1
            pbar.set_description(f"Testing Loss: {total_loss / cnt :6f}")

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0)
scheduler = ExponentialLR(optimizer, gamma=0.75)
criteria = nn.MSELoss()

from scipy.integrate import solve_ivp

def get_flow(t, x):
    with torch.no_grad():
        x = torch.from_numpy(x).to(device).float()
        x = x.reshape(-1, 3, 64, 64)
        # t = torch.from_numpy(t).to(device).unsqueeze(-1).float()
        t = torch.tensor([t]).to(device).unsqueeze(-1).float()
        return model(t, x).reshape(-1).cpu().numpy()

def generate(num=1):
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(num, 3, 64, 64).to(device)
        sol = solve_ivp(get_flow, np.array([0, 1]), x0.reshape(-1).cpu().numpy(), rtol=1e-6, atol=1e-6)
    return np.clip((sol.y[:, -1] + 1) / 2, 0, 1)