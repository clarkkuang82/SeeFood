# Part 2 — Fine-tuned ResNet-50 on Food-101


## Setup

- **Model**: `torchvision.models.resnet50` initialized from
  `IMAGENET1K_V2` pretrained weights; final `fc` replaced with
  `nn.Linear(2048, 101)`. All layers trainable (~25.6M params).
- **Data**: `torchvision.datasets.Food101` official split
  (75,750 train / 25,250 test). A 10% stratified (by permutation)
  holdout is drawn from the train split with `seed=42` for validation
  and best-checkpoint selection; final numbers are reported on the
  official 25,250-image test set.
- **Optimizer**: SGD, momentum 0.9.
- **Batch size** 64, **30 epochs**.
- Runs on Colab GPU.

## Two runs

Two training configurations were evaluated. V2 differs from V1 in two
ways: (a) a manual step learning-rate decay (10× at epoch 20, 100× at
epoch 26), and
(b) slightly stronger weight decay.

| | **V1** | **V2** |
|---|---|---|
| Learning rate | fixed `1e-3` | `1e-3` → `1e-4` @ epoch 20 → `1e-5` @ epoch 26 |
| Weight decay | `1e-4` | `5e-4` |
| Training time | ~5 h | ~5 h |

## Results

Reported on the official Food-101 test split (25,250 images).

| | **Val top-1** | **Val top-5** | **Test top-1** | **Test top-5** | Best epoch |
|---|---|---|---|---|---|
| V1 (fixed lr, lower wd) | 0.7621* | 0.9156* | **0.8695** | **0.9743** | 26 |
| V2 (manual lr decay, higher wd) | 0.8247 | 0.9479 | **0.8699** | **0.9751** | 26 |


**Trainable parameters**: 25,557,032 (same for both runs).


## Ablation: Train-Time Data Augmentation (V3)

To quantify how much of V2's accuracy is attributable to data
augmentation, we ran V3 with the **same configuration as V2 except
with all train-time augmentation removed** — train, val, and test
all use the deterministic eval transform
(`Resize(256) → CenterCrop(224) → Normalize`). Every other
hyperparameter (optimizer, lr schedule, weight decay, epochs, seed)
is identical.

| | Train top-1 (final) | Val top-1 (best) | Test top-1 | Test top-5 | Best epoch |
|---|---|---|---|---|---|
| V2 (w/ aug) | 0.820 | 0.8247 | **0.8699** | **0.9751** | 26 |
| V3 (w/o aug) | **0.998** | 0.7626 | 0.8018 | 0.9509 | 17 |
| Δ (V3 − V2) | +17.8 | -6.2 | **-6.8** | -2.4 | -9 |

V3 overfits aggressively: by epoch 9 training top-1 already exceeds
95% while validation top-1 plateaus at 76% and then slowly degrades.


## Limitations / Next steps


## Files

- `metrics.json` — raw numbers and full epoch history for the V3
  ablation run
- `training_curves.png` — loss and top-1/top-5 curves for V3
- `../checkpoints/best.pt` — V3 best checkpoint (epoch 17)

