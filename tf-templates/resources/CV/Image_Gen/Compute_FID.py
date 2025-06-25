# !pip install torcheval

from torcheval.metrics import FrechetInceptionDistance
fid = FrechetInceptionDistance(device = device)

idl = iter(train_loader)
for i in tqdm(range(8)):
    reals = ((next(idl))[0] * 1).to(device)
    # reals = (reals + 1) / 2
    fid.update(reals, True)

for i in tqdm(range(8)):
    fakes = generate_image(num_image=64)
    fakes = torch.clamp(fakes, 0, 1)
    fid.update(fakes, False)
fid.compute()