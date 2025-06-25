# Run the ALiBi algorithm directly on the model using the Composer functional API
# !pip install mosaicml

import torch
import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_alibi(
        model=model,
        max_sequence_length=1024,
    )

    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()