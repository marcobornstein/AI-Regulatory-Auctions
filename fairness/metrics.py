"""Accuracy and group-fairness metrics computed over (group, label) cells."""

import torch


class AverageMeter:
    """Running sample-weighted average."""

    def __init__(self):
        self.sum, self.count, self.avg = 0.0, 0, 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def predictions(output):
    return output.topk(1, dim=1).indices.squeeze(1)


def accuracy(output, target):
    """Top-1 accuracy in percent."""
    return predictions(output).eq(target).float().sum().mul_(100.0 / target.size(0))


def group_counts(output, target, group, num_groups=2, num_labels=2):
    """Correct predictions and sample counts per (group, label) cell."""
    correct = predictions(output).eq(target).float()
    hits, counts = torch.zeros(num_groups, num_labels), torch.zeros(num_groups, num_labels)
    for g in range(num_groups):
        for y in range(num_labels):
            mask = (group == g) & (target == y)
            hits[g, y], counts[g, y] = correct[mask].sum(), mask.sum()
    return hits, counts


def fairness_metrics(hits, counts):
    """Per-cell accuracies (acc_A{group}Y{label}, percent) and disparities between groups."""
    cell_acc = torch.nan_to_num(hits / counts) * 100
    group_acc = torch.nan_to_num(hits.sum(dim=1) / counts.sum(dim=1))  # fraction, not percent
    num_groups, num_labels = cell_acc.shape
    gap = lambda values: (values.max() - values.min()).item()
    metrics = {f"acc_A{g}Y{y}": cell_acc[g, y].item() for g in range(num_groups) for y in range(num_labels)}
    metrics.update(
        acc_var=torch.std(cell_acc, unbiased=False).item(),
        acc_dis=gap(group_acc),
        err_op_0=gap(cell_acc[:, 0]),  # equal opportunity for label 0
        err_op_1=gap(cell_acc[:, 1]),
        # Equalized odds: largest summed per-label accuracy gap over pairs of groups.
        err_odd=max((sum(abs(cell_acc[i, y] - cell_acc[j, y]) for y in range(num_labels)).item()
                     for i in range(num_groups) for j in range(i + 1, num_groups)), default=0.0),
    )
    return metrics
