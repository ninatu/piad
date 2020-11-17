import pandas as pd
import os

from piad.train import train
from piad.evaluate import evaluate
from piad.utils import load_config

# classes = list(range(100))
# coil_normal_classes = np.random.choice(classes, 30, replace=False)

# fix it how it was done in the paper
coil_normal_classes = [79, 48, 65, 98, 73, 13, 14, 39, 57, 28, 42, 75, 44, 24, 82, 56, 7, 47,
                       68, 30, 33, 90, 70, 87, 12, 91, 4, 22, 71, 55]

EXPS = [
    ('cifar10', list(range(10)), [None]),
    ('mnist', list(range(10)), [None]),
    ('fashion_mnist', list(range(10)), [None]),
    ('coil100', coil_normal_classes, [None]),
    ('celeba', [0], ["Bald", "Mustache", "Bangs", "Eyeglasses", "Wearing_Hat"]),
    ('celeba_extended', [0], ["Bald", "Mustache", "Bangs", "Eyeglasses", "Wearing_Hat"]),
    ('lsun', [0], [None]),
    ('ablation_cifar10_L2_manual_weights', list(range(10)), [None]),
    ('ablation_cifar10_L2', list(range(10)), [None]),
    ('ablation_cifar10_perc_L2', list(range(10)), [None]),
    ('ablation_cifar10_perc_L1', list(range(10)), [None]),
]

RUNS = [0, 1, 2]
BATCH_SIZE = 32

RESULTS_ROOT = './results/'

if __name__ == '__main__':
    for exp_name, normal_classes, abnormal_classes in EXPS:
        config_path = f'./configs/{exp_name}.yaml'
        config = load_config(config_path)

        roc_aucs = pd.DataFrame(
            columns=abnormal_classes if exp_name in ['celeba', 'celeba_extended'] else normal_classes,
            index=RUNS
        )

        for normal_class in normal_classes:
            for run in RUNS:

                train(config, normal_class, run)

                for abnormal_class in abnormal_classes:
                    evaluate(config, normal_class, run, abnormal_class, BATCH_SIZE)

                    if exp_name in ['celeba', 'celeba_extended']:
                        path = f'./output/{exp_name}/{normal_class}/{run}/{abnormal_class}.csv'
                        roc_auc = pd.read_csv(path)['ROC AUC'].loc[0]
                        roc_aucs[abnormal_class][run] = roc_auc
                    else:
                        path = f'./output/{exp_name}/{normal_class}/{run}/results.csv'
                        roc_auc = pd.read_csv(path)['ROC AUC'].loc[0]
                        roc_aucs[normal_class][run] = roc_auc

        roc_aucs.loc['average'] = roc_aucs.mean(axis=0)
        roc_aucs['average'] = roc_aucs.mean(axis=1)

        os.makedirs(RESULTS_ROOT, exist_ok=True)
        print(roc_aucs)
        roc_aucs.to_csv(os.path.join(RESULTS_ROOT, f"{exp_name}.csv"))
