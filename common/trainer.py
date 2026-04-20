import torch
from tqdm.auto import tqdm


def _top_k_correct(logits, targets, maxk=5):
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    c1 = correct[:1].reshape(-1).float().sum().item()
    c5 = correct[:5].reshape(-1).float().sum().item()
    return c1, c5


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_batches=None, epoch=0):
    model.train()
    total_loss = 0.0
    total_c1 = 0.0
    total_c5 = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=f"train[{epoch}]", leave=False)
    for i, (images, targets) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        c1, c5 = _top_k_correct(logits.detach(), targets)
        total_loss += loss.item() * bs
        total_c1 += c1
        total_c5 += c5
        total_n += bs
        pbar.set_postfix(loss=f"{total_loss/total_n:.3f}", top1=f"{total_c1/total_n:.3f}")

    return total_loss / total_n, total_c1 / total_n, total_c5 / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None, desc="eval"):
    model.eval()
    total_loss = 0.0
    total_c1 = 0.0
    total_c5 = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for i, (images, targets) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        bs = targets.size(0)
        c1, c5 = _top_k_correct(logits, targets)
        total_loss += loss.item() * bs
        total_c1 += c1
        total_c5 += c5
        total_n += bs

    return total_loss / total_n, total_c1 / total_n, total_c5 / total_n
