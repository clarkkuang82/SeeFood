# train_vit.py
import os
import time
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = "/home/hice1/zstern7/scratch/hf_cache"  # persist across runs
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

# ViT-B/16 expects 224x224
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        return self.transform(image), label


def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / batch_size).item())
        return results


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)}  loss: {loss.item():.4f}")
    return total_loss / len(loader)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, top1_sum, top5_sum = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += top1
            top5_sum += top5
    n = len(loader)
    return total_loss / n, top1_sum / n, top5_sum / n


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"Using device: {DEVICE}")
    

    # Load dataset — uses cache if already downloaded
    ds = load_dataset("food101", cache_dir=CACHE_DIR)

    train_ds = Food101Dataset(ds["train"],      transform_train)
    val_ds   = Food101Dataset(ds["validation"], transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 101)

    model = model.to(DEVICE)

    trainable_params = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    metrics = []
    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{EPOCHS} starting...")

        train_loss            = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, top1, top5  = val_epoch(model, val_loader, criterion)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        row = {
            "epoch":             epoch,
            "train_loss":        round(train_loss, 4),
            "val_loss":          round(val_loss, 4),
            "top1_acc":          round(top1, 4),
            "top5_acc":          round(top5, 4),
            "epoch_time_sec":    round(epoch_time, 1),
            "trainable_params":  trainable_params,
        }
        metrics.append(row)
        print(row)

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    with open("metrics.json", "w") as f:
        json.dump({"total_training_time_sec": round(total_time, 1),
                   "epochs": metrics}, f, indent=2)
    print("Saved metrics.json")

    torch.save(model.state_dict(), "vit_b16_food101.pt")
    print("Saved model checkpoint")


if __name__ == "__main__":
    main()