# CS7643 project

## Run instructions

1. Use **Python 3.12+**. From the repository root, create and activate a virtual environment, then install dependencies: `pip install -e .`
2. Install a Jupyter front end if needed, for example: `pip install notebook` (or JupyterLab).
3. Authenticate with Weights & Biases: `wandb login`, or set the `WANDB_API_KEY` environment variable before starting Jupyter.
4. Open `proj.ipynb`, then run the cells in order from top to bottom (Run All). The first run may download data or expect a dataset path configured in the notebook; adjust those cells if your machine or Colab layout differs.

## Training

The notebook trains an image classifier on [Food-101](https://huggingface.co/datasets/food101) (101 classes). The model is a **ResNet-50** initialized with **ImageNet1K_V1** weights. All backbone parameters are **frozen**; only the final fully connected layer is replaced with a `101`-way linear head and trained with **Adam** on that head only, **cross-entropy loss**, and an optional **weight decay** taken from a small hyperparameter grid (`itertools.product` over learning-rate and weight-decay lists—often a single configuration).

**Data and preprocessing.** Images are resized to `224×224`, converted to tensors, and normalized with ImageNet mean and standard deviation. The Hugging Face validation split is further split 50/50 (seed `42`): half is used as **validation**, and the official **test** split is used for a separate evaluation cell after training.

**Training loop.** For each grid configuration, the notebook builds fresh loaders (default batch size **64**), runs many **epochs** (default **100**) of `train_one_epoch` followed by validation `evaluate`. Training loss is the mean cross-entropy over batches. Validation reports **top-1 accuracy**, **top-5 accuracy**, and **mean loss**. The best validation top-1 in a run triggers a checkpoint save; the best top-1 across the whole grid updates a global “best overall” checkpoint.

## How metrics are logged

The notebook sends metrics to **[Weights & Biases](https://wandb.ai)** (`wandb`).

1. **Run start.** For each `(learning_rate, weight_decay)` in the grid, `wandb.init(project=..., name=..., config={...})` starts a new run. The run `name` encodes hyperparameters and a time stamp; `config` records `lr`, `weight_decay`, `batch_size`, and `epochs`.

2. **Per epoch.** After each epoch, `wandb.log({...}, step=epoch)` records scalars with the step set to the epoch index:
   - `train_loss` — mean training loss for that epoch  
   - `val_acc` — validation top-1 accuracy  
   - `val_top5_acc` — validation top-5 accuracy  
   - `val_loss` — mean cross-entropy on the validation set for that epoch  

3. **End of a grid configuration.** `wandb.finish()` closes the run before the notebook deletes the model and moves to the next `(lr, weight_decay)` pair.

**Console.** The notebook prints progress per epoch (for example validation top-1 and top-5). A later cell evaluates on the **test** loader and prints test metrics; those values are not written to W&B in the default notebook flow.
