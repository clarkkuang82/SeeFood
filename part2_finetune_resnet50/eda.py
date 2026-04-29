"""Quick EDA on Food-101: class count, per-class examples, image sizes."""
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main():
    from datasets import load_dataset

    ds = load_dataset("food101")
    train, test = ds["train"], ds["validation"]

    classes = train.features["label"].names
    print(f"Number of classes: {len(classes)}")
    print(f"Train split size:  {len(train):,}")
    print(f"Test split size:   {len(test):,}")
    print(f"Total examples:    {len(train) + len(test):,}")

    train_counts = Counter(train["label"])
    test_counts = Counter(test["label"])
    print(f"\nPer-class counts:")
    print(f"  train: min={min(train_counts.values())}, "
          f"max={max(train_counts.values())}, "
          f"unique sizes={len(set(train_counts.values()))}")
    print(f"  test:  min={min(test_counts.values())}, "
          f"max={max(test_counts.values())}, "
          f"unique sizes={len(set(test_counts.values()))}")

    print(f"\nFirst 10 classes: {classes[:10]}")
    print(f"Last 10 classes:  {classes[-10:]}")

    # Image size sampling (50 random train images)
    import random
    random.seed(42)
    sample_idx = random.sample(range(len(train)), 50)
    widths, heights = [], []
    for i in sample_idx:
        w, h = train[i]["image"].size
        widths.append(w)
        heights.append(h)
    print(f"\nImage sizes (50-image sample):")
    print(f"  width:  min={min(widths)} max={max(widths)} mean={sum(widths)/50:.0f}")
    print(f"  height: min={min(heights)} max={max(heights)} mean={sum(heights)/50:.0f}")
    print(f"  aspect ratios: min={min(w/h for w,h in zip(widths, heights)):.2f}, "
          f"max={max(w/h for w,h in zip(widths, heights)):.2f}")


if __name__ == "__main__":
    main()
