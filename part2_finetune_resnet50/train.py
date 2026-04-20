"""Fine-tune pretrained ResNet-50 on Food-101."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet50_Weights, resnet50

sys.path.append(str(Path(__file__).parent.parent))
from common.data import build_food101_splits, build_loaders, set_seed, get_device
from common.trainer import evaluate, train_one_epoch
from common.metrics import count_params


def build_smoke_loaders(batch_size=8, image_size=224, num_classes=101, n_train=64, n_val=16):
    g = torch.Generator().manual_seed(0)
    train_x = torch.randn(n_train, 3, image_size, image_size, generator=g)
    train_y = torch.randint(0, num_classes, (n_train,), generator=g)
    val_x = torch.randn(n_val, 3, image_size, image_size, generator=g)
    val_y = torch.randint(0, num_classes, (n_val,), generator=g)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(num_classes=101):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def plot_curves(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["train_top1"] for h in history], label="train top-1")
    axes[1].plot(epochs, [h["val_top1"] for h in history], label="val top-1")
    axes[1].plot(epochs, [h["val_top5"] for h in history], label="val top-5", linestyle="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny smoke-test on synthetic data (no download).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    output_dir = Path(cfg["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    # Data
    if args.smoke:
        print("[smoke] using random synthetic data — no Food-101 download.")
        train_loader, val_loader = build_smoke_loaders(
            batch_size=8, image_size=cfg["image_size"],
        )
    else:
        train, val, test = build_food101_splits(
            root=cfg["data_root"], image_size=cfg["image_size"], download=True,
        )
        train_loader, val_loader, _ = build_loaders(
            train, val, test, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            pin_memory=device.type == "cuda",
        )

    # Model
    model = build_model(num_classes=101).to(device)
    params = count_params(model)

    optimizer = SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                    weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
    criterion = nn.CrossEntropyLoss()
    epochs = 2 if args.smoke else cfg["epochs"]
    max_batches = 2 if args.smoke else None

    # Train
    history = []
    best_top1 = -1.0
    best_epoch = -1
    start = time.time()
    ckpt_path = output_dir / "checkpoints" / "best.pt"
    for epoch in range(epochs):
        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_batches=max_batches, epoch=epoch,
        )
        va_loss, va_top1, va_top5 = evaluate(
            model, val_loader, criterion, device,
            max_batches=max_batches, desc=f"val[{epoch}]",
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_top1": tr_top1, "train_top5": tr_top5,
            "val_loss": va_loss, "val_top1": va_top1, "val_top5": va_top5,
        })
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} top1={tr_top1:.4f} | "
              f"val_loss={va_loss:.4f} top1={va_top1:.4f} top5={va_top5:.4f}")

        if va_top1 > best_top1:
            best_top1 = va_top1
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_top1": va_top1}, ckpt_path)

    elapsed = time.time() - start

    if args.smoke:
        for h in history:
            assert not math.isnan(h["train_loss"]), "train loss is NaN"
            assert not math.isinf(h["train_loss"]), "train loss is Inf"

    # Save artifacts
    best = max(history, key=lambda h: h["val_top1"])
    metrics = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "trainable_params": params["trainable_params"],
        "total_params": params["total_params"],
        "training_time_sec": elapsed,
        "best_epoch": best_epoch,
        "val_top1": best["val_top1"], "val_top5": best["val_top5"], "val_loss": best["val_loss"],
        "test_top1": None, "test_top5": None, "test_loss": None,
        "history": history,
    }
    with open(output_dir / "results" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_curves(history, output_dir / "results" / "training_curves.png")
    print(f"Done. best val top-1={best_top1:.4f} @ epoch {best_epoch}. "
          f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
