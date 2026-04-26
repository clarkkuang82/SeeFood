"""Evaluate a fine-tuned checkpoint on the Food-101 test split."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from common.data import (build_food101_splits, build_food101_splits_hf,
                         build_loaders, get_device)
from common.trainer import evaluate
from part2_finetune_resnet50.train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    output_dir = Path(cfg["output_dir"])

    pipeline = cfg.get("data_pipeline", "torchvision")
    if pipeline == "hf":
        train, val, test = build_food101_splits_hf(
            image_size=cfg["image_size"], cache_dir=cfg.get("hf_cache_dir"),
            train_aug=cfg.get("train_aug", False),
        )
    else:
        train, val, test = build_food101_splits(
            root=cfg["data_root"], image_size=cfg["image_size"], download=False,
        )
    _, _, test_loader = build_loaders(
        train, val, test, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name=cfg.get("model_name", "resnet50"), num_classes=101).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    criterion = nn.CrossEntropyLoss()

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion, device, desc="test")
    print(f"test: loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")

    metrics_path = output_dir / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {"run_name": cfg["run_name"]}
    metrics["test_loss"] = test_loss
    metrics["test_top1"] = test_top1
    metrics["test_top5"] = test_top5
    metrics["eval_ckpt"] = str(args.ckpt)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"updated {metrics_path}")


if __name__ == "__main__":
    main()
