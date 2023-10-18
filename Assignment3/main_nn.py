import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    np.random.seed(0)
    np.random.RandomState(0)

    # X, y = make_classification(n_samples=100000, n_features=20, n_redundant=2, n_classes=2)
    # data_dir = '/Users/dermacintosh/Desktop/OMSCS/Fall2023/CS7641'
    if 'Assignment3' not in os.getcwd():
        os.chdir('Assignment3')

    # data_set = sys.argv[1]
    data_set = 'tic'
    if data_set == 'tic':
        data_dir = './tic+tac+toe+endgame/'
        df = pd.read_csv(os.path.join(data_dir, 'tic-tac-toe.data'), header=None).sample(frac=1)
        X, y = df.iloc[:, :-1].replace(['x', 'o', 'b'], [2, 1, 0]), df.iloc[:, -1].replace(['negative', 'positive'],
                                                                                           [0, 1])
        os.chdir(data_dir)
    elif data_set == 'diag':
        data_dir = './breast+cancer+wisconsin+diagnostic/'
        df = pd.read_csv(os.path.join(data_dir, 'wdbc.data'), header=None).sample(frac=1)
        X, y = df.iloc[:, 2:], df.iloc[:, 1].replace(['M', 'B'], [1, 0])
        os.chdir(data_dir)
    else:
        raise ValueError('Incorrect dataset specified, please run python main.py tic or python main.py diag')

    # Per dataset
    # 1. RUN 2 clusters,
    # 2. RUN 4 dim reduce
    # 3.   RUN 2 clusters on dim reduced data (8 combos)
    #      SHOW one cluster+linear and cluster+manifold (2 comparisons) 2 plots per dataset

    # Pick one dataset
    # 4. RUN MLP on 4 dim reduced data (show one linear vs manifold) 2 plots
    # 5. RUN MLP on 2 clusters (show both clustering techniques) 2 plots

    #run_dict = dict(KMeans=dict(),
    #                EM=dict())

    param_grid = dict(hidden_layer_sizes=[1, 2, 5, 10, 25, 50, 100, 250, 500],
                      activation= ['identity', 'logistic', 'relu'],
                      learning_rate=[1E-1, 1E-2, 1E-3, 1E-4, 1E-5])


