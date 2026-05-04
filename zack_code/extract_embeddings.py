# extract_embeddings.py
import os
import torch
import json
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = "/home/hice1/zstern7/scratch/hf_cache"
CHECKPOINT  = "/home/hice1/zstern7/repos/dl_a5/vit_b16_full_food101.pt"
BATCH_SIZE  = 128
NUM_WORKERS = 8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────
print('at least were here')
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


def main():
    print(f"Using device: {DEVICE}")

    # Load dataset
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    ds = load_dataset("food101", cache_dir=CACHE_DIR)

    # Save label names
    label_names = ds["validation"].features["label"].names
    with open("label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"Saved {len(label_names)} label names")

    val_ds = Food101Dataset(ds["validation"], transform_val)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)

    # Load model and replace head with Identity
    model = vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 101)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.heads.head = torch.nn.Identity()  # extract 768-dim embeddings
    model = model.to(DEVICE)
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            if i % 10 == 0:
                print(f"  batch {i}/{len(val_loader)}")

    all_embeddings = torch.cat(all_embeddings).numpy()  # (25250, 768)
    all_labels = torch.cat(all_labels).numpy()           # (25250,)

    print(f"Embeddings shape: {all_embeddings.shape}")

    np.save("/home/hice1/zstern7/repos/dl_a5/embeddings/embeddings.npy", all_embeddings)
    np.save("/home/hice1/zstern7/repos/dl_a5/embeddings/labels.npy", all_labels)
    print("Saved embeddings.npy and labels.npy")


if __name__ == "__main__":
    main()