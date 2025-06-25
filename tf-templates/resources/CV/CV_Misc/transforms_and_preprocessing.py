from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

import torch.backends.cudnn as cudnn
cudnn.benchmark = True # Remember to add this line!!!!

to_tensor_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Resize((128, 128)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Img_Dataset(Dataset):
    def __init__(self, filepaths, labels, transforms=None):
        self.transforms = transforms
        self.labels = labels
        self.imgs = filepaths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = to_tensor_transform(Image.open(self.imgs[idx]))
        if self.transforms is None:
            return img, self.labels[idx]
        return self.transforms(img), self.labels[idx]
    
train_transform = v2.Compose([
    v2.TrivialAugmentWide(),
    # v2.RandomRotation(15),
    # v2.RandomAffine(45),
    # v2.ElasticTransform()
])
train_ds = Img_Dataset(train_image_paths, train_image_labels, train_transform)
train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    num_workers=4,
)

# scaler = torch.amp.GradScaler("cuda")
def train(model, criteria, optimiser, scheduler=None):
    model.train()
    total_loss = 0.0
    itr_cnt = 0
    num_correct = 0
    cnt = 0
    cutmix = v2.CutMix(num_classes=len(classes))
    mixup = v2.MixUp(num_classes=len(classes))
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    for imgs, labels in (pbar := tqdm(train_loader)):
        # with torch.autocast(device_type=device, dtype=torch.float16):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = cutmix_or_mixup(imgs, labels)
        output = model(imgs)
        loss = criteria(output, labels)
        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimiser)
        # scaler.update()
        optimiser.step()
        optimiser.zero_grad()
        total_loss += loss.item()
        itr_cnt += 1
        cnt += labels.shape[0]
        # num_correct += sum(output.argmax(-1) == labels)
        num_correct += sum(output.argmax(-1) == labels.argmax(-1))
        pbar.set_description(f"Average Loss: {total_loss / itr_cnt :6f} | Train Accuracy: {num_correct / cnt * 100 :6f}")
    if scheduler is not None:
        scheduler.step()

def test(model, criteria, optimiser=None, scheduler=None):
    model.eval()
    total_loss = 0.0
    itr_cnt = 0
    num_correct = 0
    cnt = 0
    with torch.no_grad():
        for imgs, labels in (pbar := tqdm(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criteria(output, labels)
            total_loss += loss.item()
            itr_cnt += 1
            cnt += labels.shape[0]
            num_correct += sum(output.argmax(-1) == labels)
            pbar.set_description(f"Average Loss: {total_loss / itr_cnt :6f} | Test Accuracy: {num_correct / cnt * 100 :6f}")