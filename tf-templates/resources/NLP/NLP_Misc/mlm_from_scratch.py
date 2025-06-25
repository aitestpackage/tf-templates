def get_masked(ids, mlm_prob=0.15):
    # ids: input tokens
    # mlm_prob: mask prob

    # generate tokens mask
    tokens_to_mask = torch.rand(ids.shape) < mlm_prob
    tokens_to_mask = tokens_to_mask.to(torch.int32).to(device)

    # generate the new input
    # unfortunately cannot slice or view so have to use a work-around
    new_ids = ids.detach().clone()
    new_ids = new_ids * (1 - tokens_to_mask) + mask_token_id * tokens_to_mask
    return new_ids, tokens_to_mask

def pretrain(pretrain_model, mlm_criteria, mlm_optimiser, mlm_scheduler=None):
    pretrain_model.train()
    total_loss = 0
    for cnt, batch in (pbar := tqdm(enumerate(train_loader))):
        input_ids = batch[0]
        target = batch[1]
        new_input_ids, mask = get_masked(input_ids, 0.40)
        mask = mask.to(torch.bool)
        mlm_optimiser.zero_grad()
        # pass in the masked ids
        output = pretrain_model(new_input_ids)

        loss1 = mlm_criteria(output.reshape(-1, vocab_size), input_ids.reshape(-1)) 
        # the model should predict the given tokens well enough

        # masking the MLM part (prediction of [MASK])
        output = torch.masked_select(output, mask.unsqueeze(-1)).reshape(-1, vocab_size)
        target = torch.masked_select(input_ids, mask)
        # print(input_ids.shape, output.shape, target.shape)
        # break
        loss = mlm_criteria(output, target)

        # Add the loss
        loss = loss + loss1 * 0.25
        loss.backward()
        mlm_optimiser.step()
        total_loss += loss.item()
        pbar.set_description(f"Average Loss: {total_loss / (cnt + 1):.6f}")
    if mlm_scheduler is not None:
        mlm_scheduler.step()