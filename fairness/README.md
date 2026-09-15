# Compliance-cost case study: fairness on FairFace

Circa assumes that higher compliance costs more (Assumption 2 in the paper). This case study
tests that assumption for one concrete notion of compliance, **equalized odds**. Equalized odds
measures whether groups have similar true-positive and false-positive rates; lower is fairer. The
cost of better compliance here is collecting more minority-group training data.

## Setup

- **Task:** gender classification on [FairFace](https://github.com/joojs/fairface) (Kärkkäinen and Joo, 2021), with race (White vs. Black) as the sensitive attribute.
- **Model:** VGG-16 trained from scratch for 50 epochs. Training uses SGD with learning rate 0.001, momentum 0.9, weight decay 5e-4, and batch size 128.
- **Data:** the training set has 5,000 White images plus Black images making up 5% to 50% of the set. The validation set is built from 500 White images with the same mix. The test set is balanced: 1,500 images across the four race × gender cells.
- **Reported metric:** test equalized odds at the epoch with the best validation accuracy, averaged over ten seeds.

| Minority Class % | Mean Equalized Odds Score |
|------------------|---------------------------|
| 5%               | 22.55                     |
| 10%              | 22.31                     |
| 15%              | 18.97                     |
| 20%              | 17.46                     |
| 25%              | 15.78                     |
| 30%              | 15.44                     |
| 35%              | 13.09                     |
| 40%              | 11.01                     |
| 45%              | 9.83                      |
| 50%              | 9.38                      |

The per-run CSVs behind this table are in [`results/`](results). `pytest` checks that the table matches them.

## Data

The CSVs in `data/fairface/` list the White and Black subset of FairFace used in the paper, with
race coded as 0 = White and 1 = Black. The FairFace images themselves are not included. Download
`fairface-img-margin025-trainval.zip` (padding 0.25) from the
[FairFace repository](https://github.com/joojs/fairface) and unzip it inside `data/fairface/`, so
that paths such as `data/fairface/train/1.jpg` exist. FairFace is licensed under CC BY 4.0.

## Training

From the repository root, run `pip install -e ".[fairness]"`. Then, to reproduce all 100 runs:

```bash
cd fairness
for seed in $(seq 1 10); do
  for p in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python main.py --per-min $p --seed $seed
  done
done
```

Training uses CUDA when it is available. Run `python main.py --help` to see all options.

Each run writes `results/<num-maj>-<per-min>-<seed>.csv`. The file has one row per validation
epoch, followed by two test rows: one at the epoch with the best validation accuracy − equalized
odds, and one at the epoch with the best validation accuracy. The columns are overall accuracy,
per-cell accuracy `acc_A{race}Y{gender}`, the spread of the cell accuracies (`acc_var`), the gap in
overall group accuracy (`acc_dis`), equal-opportunity gaps (`err_op_0`, `err_op_1`), and
equalized odds (`err_odd`).

## Figure 5

```bash
python ablation.py
```

This prints the table above and saves `../figures/fairness_ablation.png`. The figure plots cost
(minority share) against safety (inverse equalized odds), both normalized to a maximum of 1,
together with a quadratic fit of the price-of-safety curve $M$.
