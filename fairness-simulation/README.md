We conduct an ablation study to demonstrate that in realistic settings, safety is mapped to cost in a monotonically increasing way (as detailed in [Assumption 2](https://arxiv.org/abs/2410.01871)).

While there are many factors to consider when gauging safe AI deployment, we analyze model fairness, via equalized odds, for image classification in this study. Equalized odds measures if different groups have similar true positive rates and false positive rates (lower is better).



### Experiment Set up
- We train VGG-16 models on the Fairface dataset [Kärkkäinen 2019] for 50 epochs (repeated ten times with different random seeds), and consider a gender classification task with race as the sensitive attribute.
- Models with the largest validation classification accuracy during training are selected for testing.
- Many types of costs exist for training safer models, such as extensive architecture and hyper-parameter search. In this study, we consider the cost of an agent acquiring more minority class data.
- This leads to a larger and more balanced dataset. We simulate various mixtures of training data, starting from a 95:5 skew and scaling up to fully balanced training data with respect to the sensitive attribute. In our study, we gauge equalized odds performance on well-balanced test data for the models trained on various mixtures of data. Below we tabulate our results.

| Minority Class % | Mean Equalized Odds Score |
|-----------------|---------------------------|
| 5%              | 22.55                    |
| 10%             | 22.31                    |
| 15%             | 18.97                    |
| 20%             | 17.46                    |
| 25%             | 15.78                    |
| 30%             | 15.44                    |
| 35%             | 13.09                    |
| 40%             | 11.01                    |
| 45%             | 9.83                     |
| 50%             | 9.38                     |

### Training and Evaluation
```
python main.py \
    --num-maj 1000 \     # Number of majority class samples
    --per-min 0.3 \      # Percentage of minority class (0.0-1.0)
    --epochs 50 \
    --batch-size 128 \
    --lr 0.001 \
    --fair-type "eql_odd"
```