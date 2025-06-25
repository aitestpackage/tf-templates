# !pip install mosaicml
import composer.functional as cf
# cf.__all__
# help(cf.apply_gyro_dropout)

cf.apply_blurpool(
    resnet_model,
    replace_convs=True,
    replace_maxpools=True,
    blur_first=True
)
cf.apply_gyro_dropout(
    model,
    iters_per_epoch = 196,
    max_epoch = 100,
    p = 0.5,
    sigma = 256,
    tau = 16,
)
cf.apply_ghost_batchnorm(
    model,
    ghost_batch_size=32,
    optimizers=opt
)
def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_resized, _ = cf.resize_batch(X, y, scale_factor=0.5, mode='resize', resize_targets=False)
            y_hat = model(X_resized)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()