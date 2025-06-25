class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=False):
        super().__init__()
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=stride, padding=1)
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=stride, padding=1)
            self.skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        self.gn1 = nn.BatchNorm2d(out_channels)
        self.gn2 = nn.BatchNorm2d(out_channels)
        self.gn3 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x1 = F.silu(self.gn1(self.conv1(x)))
        x1 = F.silu(self.gn2(self.conv2(x1)))
        x = self.skip(x) + x1
        x = self.gn3(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_in = nn.Linear(128, 100 * 1 * 1)
        self.resnet = nn.Sequential(
            nn.ConvTranspose2d(100, 256, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.Dropout(0.25),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(),
            ResNetBlock(256, 256, 2, upsample=True),
            nn.Dropout(0.25),
            ResNetBlock(256, 128, 2, upsample=True),
            nn.Dropout(0.25),
            ResNetBlock(128, 96, 2, upsample=True),
            nn.Dropout(0.25),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(96, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        )
    def forward(self, x):
        x = x.squeeze()
        x = self.to_in(x)
        x = x.reshape(-1, 100, 1, 1)
        x = self.resnet(x)
        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.25),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
            # nn.Conv2d(3, 128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.25),
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.25),
            nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.25),
            nn.Conv2d(512, 1, kernel_size=4, padding=0, stride=1),
        )
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

def train():
    discriminator.train()
    generator.train()
    cnt = 0
    cnt2 = 0
    t_d_loss = 0
    t_g_loss = 0
    for x in (pbar := tqdm(train_loader)):
        optimizerD.zero_grad()
        cnt += 1
        
        img = x[0].to(device)
        img = img * 2 - 1
        
        # Discriminator Update
        # Generate all fake batch WITHOUT grad
        noise = torch.randn(img.shape[0], 128, 1, 1, device=device)
        with torch.no_grad():
            fake_img = generator(noise)
        instance_noise = torch.randn(img.shape, device=device) * 0.01
        real_d = discriminator(img + instance_noise)
        fake_d = discriminator(fake_img + instance_noise)
        loss_d = -torch.mean((real_d))
        loss_d = loss_d + torch.mean((fake_d))
        
        # L2 Norm Loss instead of weight clipping
        alpha = torch.rand(img.shape[0], 1, 1, 1, device=device)
        new_img = (alpha * img + (1 - alpha) * fake_img.detach()) # Similar to flow matching
        new_img = Variable(new_img, requires_grad=True)
        d_new = discriminator(new_img)
        
        # Compute gradient
        d_grad = torch.autograd.grad(
            d_new,
            new_img,
            grad_outputs=torch.ones(d_new.shape, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        d_grad = d_grad.reshape(img.shape[0], -1)
        loss_d = loss_d + 10 * torch.mean(torch.square((torch.norm(d_grad, 2, dim=-1) - 2.0)))

        t_d_loss += loss_d.item()
        # Backprop Discriminator
        loss_d.backward()
        optimizerD.step()

        # Generator Update
        if cnt % n_critic == 0:
            optimizerG.zero_grad()
            # Generate all fake batch WITH grad
            noise = torch.randn(img.shape[0], 128, 1, 1, device=device)
            fake_img = generator(noise)
            loss_g = -torch.mean(discriminator(fake_img + instance_noise))
            t_g_loss += loss_g.item()
            loss_g.backward()
            optimizerG.step()
            cnt2 += 1
            # Update tqdm
            pbar.set_description(f"G Loss: {t_g_loss / cnt2 :6f} | D Loss: {t_d_loss / cnt :6f}")
    if schedulerD is not None:
        schedulerD.step()
    if schedulerG is not None:
        schedulerG.step()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.005)
        nn.init.constant_(m.bias.data, 0)
generator.apply(weights_init)
discriminator.apply(weights_init);


from torch.nn.utils import spectral_norm
def apply_spectral_norm(model):
    """
    Recursively wraps all Conv2d and Linear layers in `model`
    with spectral_norm, in-place.
    """
    for name, module in model.named_children():
        # If this module itself has children, recurse into it
        if len(list(module.children())) > 0:
            apply_spectral_norm(module)
        # Otherwise, if it's Conv2d or Linear, wrap it
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            # Replace the module in-place
            wrapped = spectral_norm(module)
            setattr(model, name, wrapped)

apply_spectral_norm(discriminator)
discriminator = discriminator.to(device)

optimizerD = SGD(discriminator.parameters(), lr=3e-4)
schedulerD = ExponentialLR(optimizerD, gamma=0.99)
optimizerG = RMSprop(generator.parameters(), lr=3e-4)
schedulerG = ExponentialLR(optimizerG, gamma=0.99)

n_critic = 5

for i in range(3):
    train()
    test()
