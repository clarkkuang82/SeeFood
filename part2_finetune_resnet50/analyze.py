"""Error analysis on a trained checkpoint:
confusion matrix, per-class accuracy, top confused pairs, misclassified examples."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from common.data import (build_food101_splits, build_food101_splits_hf,
                         build_loaders, get_device, set_seed)
from part2_finetune_resnet50.train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default=None,
                        help="Defaults to <output_dir>/results/analysis/")
    parser.add_argument("--n_misclassified", type=int, default=12)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    device = get_device()

    out_dir = Path(args.out_dir or Path(cfg["output_dir"]) / "results" / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build data with the same pipeline as training, but with eval transform on train
    pipeline = cfg.get("data_pipeline", "torchvision")
    if pipeline == "hf":
        train, val, test = build_food101_splits_hf(
            image_size=cfg["image_size"], cache_dir=cfg.get("hf_cache_dir"),
            train_aug=False,
        )
    else:
        train, val, test = build_food101_splits(
            root=cfg["data_root"], image_size=cfg["image_size"], download=False,
        )
    _, _, test_loader = build_loaders(
        train, val, test, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
    )

    # Class names — pull from whichever pipeline we are using
    if pipeline == "hf":
        from datasets import load_dataset
        classes = load_dataset("food101", cache_dir=cfg.get("hf_cache_dir"),
                               split="train").features["label"].names
    else:
        from torchvision.datasets import Food101
        classes = Food101(root=cfg["data_root"], split="test", download=False).classes

    model = build_model(model_name=cfg.get("model_name", "resnet50"),
                        num_classes=len(classes)).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # Inference: collect predictions, labels, top-1 confidences
    all_preds, all_labels, all_confs = [], [], []
    misclassified = []  # (test_index, true, pred, conf)
    with torch.no_grad():
        idx = 0
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top_conf, pred = probs.max(dim=1)
            pred_cpu = pred.cpu().numpy()
            tgt_cpu = targets.numpy()
            conf_cpu = top_conf.cpu().numpy()
            all_preds.append(pred_cpu)
            all_labels.append(tgt_cpu)
            all_confs.append(conf_cpu)
            for j in range(len(tgt_cpu)):
                if pred_cpu[j] != tgt_cpu[j]:
                    misclassified.append((idx + j, int(tgt_cpu[j]),
                                          int(pred_cpu[j]), float(conf_cpu[j])))
            idx += len(tgt_cpu)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # 1. Confusion matrix (101 x 101) — row-normalized so each row sums to 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, cmap="viridis", cbar=True, ax=ax,
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized) — {Path(args.ckpt).name}")
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Per-class accuracy (diagonal of normalized cm)
    per_class_acc = np.diag(cm_norm)
    order = np.argsort(per_class_acc)
    bottom5 = [(classes[i], float(per_class_acc[i])) for i in order[:5]]
    top5 = [(classes[i], float(per_class_acc[i])) for i in order[-5:][::-1]]

    with open(out_dir / "per_class_accuracy.csv", "w") as f:
        f.write("class,accuracy\n")
        for i in np.argsort(-per_class_acc):
            f.write(f"{classes[i]},{per_class_acc[i]:.4f}\n")

    fig, ax = plt.subplots(figsize=(20, 5))
    sorted_idx = np.argsort(-per_class_acc)
    ax.bar(range(len(classes)), per_class_acc[sorted_idx], width=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in sorted_idx], rotation=90, fontsize=5)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlim(-1, len(classes))
    ax.set_title("Per-class top-1 accuracy (sorted)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_accuracy.png", dpi=180)
    plt.close()

    # 3. Top confused pairs — largest off-diagonal entries
    cm_off = cm.copy().astype(np.int64)
    np.fill_diagonal(cm_off, 0)
    flat_idx = np.argsort(-cm_off, axis=None)[:15]
    pair_lines = ["true -> predicted : count (rate)"]
    for fi in flat_idx:
        t, p = np.unravel_index(fi, cm_off.shape)
        if cm_off[t, p] == 0:
            break
        rate = cm_norm[t, p]
        pair_lines.append(f"{classes[t]} -> {classes[p]} : {cm_off[t,p]} ({rate:.1%})")
    (out_dir / "top_confused_pairs.txt").write_text("\n".join(pair_lines))

    # 4. Misclassified examples — pick highest-confidence wrong predictions
    misclassified.sort(key=lambda x: -x[3])
    picks = misclassified[: args.n_misclassified]

    test_ds = test_loader.dataset
    rows = (args.n_misclassified + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    axes = axes.flatten()
    for ax, (idx, t, p, conf) in zip(axes, picks):
        img, _ = test_ds[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f"true: {classes[t]}\npred: {classes[p]} ({conf:.2f})", fontsize=9)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "misclassified_examples.png", dpi=140)
    plt.close()

    summary = {
        "ckpt": str(args.ckpt),
        "n_test": int(len(y_true)),
        "overall_top1": float((y_pred == y_true).mean()),
        "top5_classes": top5,
        "bottom5_classes": bottom5,
        "n_misclassified_total": int((y_pred != y_true).sum()),
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
