"""Train VGG-16 on a FairFace mixture with a given minority share and record accuracy and fairness metrics.

Training loop adapted from "Transferring Fairness under Distribution Shifts via Fair Consistency
Regularization" by Bang An, Zora Che, Mucong Ding, and Furong Huang
(https://github.com/umd-huang-lab/transfer-fairness).
"""

import argparse
import csv
import random
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FairFaceDataset, sample_mixture
from metrics import AverageMeter, accuracy, fairness_metrics, group_counts
from model import FaceClassifier

HERE = Path(__file__).resolve().parent
VAL_NUM_MAJORITY = 500
FIELDS = ["name", "epoch", "acc", "acc_A0Y0", "acc_A0Y1", "acc_A1Y0", "acc_A1Y1",
          "acc_var", "acc_dis", "err_op_0", "err_op_1", "err_odd"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-maj", type=int, default=5000, help="number of majority-group (White) training images")
    parser.add_argument("--per-min", type=float, default=0.2, help="minority-group (Black) share of train and val")
    parser.add_argument("--epochs", "--epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--step-lr", type=int, default=100, help="StepLR period in epochs")
    parser.add_argument("--step-lr-gamma", type=float, default=0.1)
    parser.add_argument("--val-epoch", type=int, default=1, help="validate every this many epochs")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--root", type=Path, default=HERE / "data" / "fairface", help="FairFace directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-csv-path", type=Path, default=HERE / "results")
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    """One pass over loader, training if an optimizer is given. Returns loss, accuracy, and fairness metrics."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    losses, accs = AverageMeter(), AverageMeter()
    hits, counts = torch.zeros(2, 2), torch.zeros(2, 2)
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"]["gender"].to(device)
            groups = batch["label"]["race"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.update(loss.item(), inputs.size(0))
            accs.update(accuracy(outputs, labels).item(), inputs.size(0))
            batch_hits, batch_counts = group_counts(outputs, labels, groups)
            hits += batch_hits
            counts += batch_counts
    return {"loss": losses.avg, "acc": accs.avg, **fairness_metrics(hits, counts)}


def log(split, epoch, metrics):
    print(f"{split} {epoch} | loss {metrics['loss']:.4f} | acc {metrics['acc']:.2f} | "
          f"acc_var {metrics['acc_var']:.2f} | err_op_0 {metrics['err_op_0']:.2f} | "
          f"err_op_1 {metrics['err_op_1']:.2f} | err_odd {metrics['err_odd']:.2f}")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    name = f"{args.num_maj}-{args.per_min}-{args.seed}"
    print(args)

    frames = {
        "train": sample_mixture(args.root / "train_white_black.csv", args.num_maj, args.per_min, args.seed),
        "val": sample_mixture(args.root / "val_white_black.csv", VAL_NUM_MAJORITY, args.per_min, args.seed),
        "test": pd.read_csv(args.root / "test_white_black.csv"),
    }
    # Evaluation loaders also shuffle, as in the runs that produced results/: this keeps the torch RNG stream identical.
    loaders = {
        split: DataLoader(FairFaceDataset(str(args.root), frame, transforms.ToTensor()), shuffle=True,
                          batch_size=args.test_batch_size if split == "test" else args.batch_size,
                          num_workers=args.num_workers)
        for split, frame in frames.items()
    }
    for split, frame in frames.items():
        print(f"{split} size: {len(frame)}")

    model = FaceClassifier(num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)
    row = lambda label, metrics: [name, label] + [metrics[field] for field in FIELDS[2:]]

    # Test metrics are recorded at the best validation (accuracy - equalized odds) and the best validation accuracy.
    rows, best_fair, best_acc, test_at_best_fair, test_at_best_acc = [], 0.0, 0.0, None, None
    for epoch in range(1, args.epochs + 1):
        log("Train", epoch, run_epoch(model, loaders["train"], device, optimizer))
        if epoch % args.val_epoch == 0:
            val = run_epoch(model, loaders["val"], device)
            log("Val", epoch, val)
            rows.append(row(f"Val {epoch}", val))
            fair_score = torch.tensor(val["acc"]) - val["err_odd"]  # float32, matching the published runs' tie-breaking
            fair_improved = bool(fair_score >= best_fair)
            if fair_improved:
                best_fair = fair_score
                test_at_best_fair = row(f"Test {epoch}", run_epoch(model, loaders["test"], device))
            if val["acc"] >= best_acc:
                best_acc = val["acc"]
                test_at_best_acc = (test_at_best_fair if fair_improved
                                    else row(f"Test {epoch}", run_epoch(model, loaders["test"], device)))
        scheduler.step()

    if test_at_best_fair is None or test_at_best_acc is None:
        raise RuntimeError("no test evaluation was recorded; check --epochs and --val-epoch")
    args.save_csv_path.mkdir(parents=True, exist_ok=True)
    with open(args.save_csv_path / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows + [test_at_best_fair, test_at_best_acc])


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
