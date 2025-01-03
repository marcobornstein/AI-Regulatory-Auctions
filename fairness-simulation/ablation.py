import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

    data_path = 'results/iclr'
    filenames = os.listdir(data_path)
    csv_files = [filename for filename in filenames if filename.endswith(".csv")]
    num_experiments = np.count_nonzero([1 if int(f.split('-')[-1].split('.')[0]) == 1 else 0 for f in csv_files])
    num_runs = 10
    step = 0.05
    pd.set_option('display.max_columns', None)

    num_files = len(csv_files)
    metrics = np.empty(shape=(num_experiments, 10, num_runs), dtype=float)
    cost_axis = np.empty(shape=(num_experiments,), dtype=float)

    for csv_file in csv_files:

        csv_info = csv_file.split('-')
        run = int(csv_info[-1].split('.')[0]) - 1
        minority_class_pct = float(csv_info[-2])
        idx = int(minority_class_pct / (step - 1e-4)) - 1
        cost_axis[idx] = minority_class_pct
        result = pd.read_csv(os.path.join(data_path, csv_file))
        columns = result.columns

        # get metrics from best test acc
        metrics[idx, :, run] = result.iloc[-1, 2:]

    # generate dataframe and choose metric
    metrics = np.mean(metrics, axis=2)
    metrics = pd.DataFrame(metrics, index=cost_axis, columns=columns[2:])
    safety_metric = metrics['err_odd']

    # normalize and scale to get correct bounds for M : Safety
    safety_metric = np.power(safety_metric, -1)
    safety_metric /= np.max(safety_metric)
    cost_axis /= np.max(cost_axis)

    # best fit line
    z = np.polyfit(safety_metric, cost_axis, 2)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(np.min(safety_metric), np.max(safety_metric), 50)
    y_new = f(x_new)

    plt.figure()
    plt.scatter(safety_metric, cost_axis)
    plt.plot(x_new, y_new, 'r', label='$M$: Price of Safety')
    plt.xlabel('Safety Level $\epsilon$', fontsize="15", fontweight='bold')
    plt.ylabel('Cost', fontsize="15", fontweight='bold')
    plt.legend(loc='best', fontsize="15")
    plt.savefig('err_odd_fairness_ablation.jpg', dpi=500)
    plt.show()

