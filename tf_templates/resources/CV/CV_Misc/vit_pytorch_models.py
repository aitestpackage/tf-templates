# !pip install vit-pytorch
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)

import torch
from vit_pytorch.mobile_vit import MobileViT

mbvit_xs = MobileViT(
    image_size = (256, 256),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes = 1000
)

img = torch.randn(1, 3, 256, 256)

pred = mbvit_xs(img) # (1, 1000)

import torch
from vit_pytorch import ViT
from vit_pytorch.simmim import SimMIM

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mim = SimMIM(
    encoder = v,
    masking_ratio = 0.5  # they found 50% to yield the best results
)

images = torch.randn(8, 3, 256, 256)

loss = mim(images)
loss.backward()

# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

torch.save(v.state_dict(), './trained-vit.pt')