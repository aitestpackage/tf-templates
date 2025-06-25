# Apply surgery on the model to swap the feed-forward block
# for a gated feed-forward block using the Composer Functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_gated_linear_units(model)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for X, y in train_loader:
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()