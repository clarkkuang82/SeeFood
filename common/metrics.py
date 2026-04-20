import torch


@torch.no_grad()
def top_k_accuracy(logits, targets, ks=(1, 5)):
    maxk = max(ks)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        out[k] = correct_k / batch_size
    return out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
