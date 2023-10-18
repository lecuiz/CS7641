import os
import mlrose_hiive as mlrose
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from mlrose_hiive.runners import SKMLPRunner, NNGSRunner
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def plot_learning_curve(clf, X, y, algo):

    # fig, ax = plt.subplots()
    # fig.set_figheight(4)
    train_sizes = np.linspace(0.1, 1.0, 5)
    lc = learning_curve(clf, X, y,
                        train_sizes=train_sizes, return_times=True,
                        # cv=StratifiedKFold(n_splits=2, shuffle=True),
                        scoring='f1', shuffle=True)
    train_size, train_scores, test_scores, fit_times, score_times = lc

    oname = '_'.join([algo, 'learning_curve']) # + '.png'
    plot_from_df(train_scores, test_scores, train_size, 'F1 Score', oname)

    oname = '_'.join([algo, 'training_time']) # + '.png'
    plot_from_df(fit_times, score_times, train_size, 'Time (s)', oname)

    # oname = '_'.join([algo, 'epochs'])  # + '.png'
    # plot_epochs(clf.loss_curve_, clf.validation_scores_, oname)

    return train_scores, test_scores, fit_times, score_times

def plot_from_df(train, test, train_size, ylabel, save_name):
    fig, ax = plt.subplots()
    fig.set_figheight(3.5)

    # Test/Train plot
    train_df = pd.DataFrame([train.mean(axis=1), train.std(axis=1)],
                            index=['Training', 'StDev'], columns=train_size).T
    test_df = pd.DataFrame([test.mean(axis=1), test.std(axis=1)],
                           index=['Testing', 'StDev'], columns=train_size).T

    # train_df.plot(ax=ax, style='-o', yerr='StDev')
    # test_df.plot(ax=ax, style='-o', yerr='StDev')

    train_df['Training'].plot(ax=ax, style='-o')
    ax.fill_between(train_df.index, train_df['Training'] - train_df['StDev'], train_df['Training'] + train_df['StDev'], alpha=0.2)
    test_df['Testing'].plot(ax=ax, style='-o')
    ax.fill_between(test_df.index, test_df['Testing'] - test_df['StDev'], test_df['Testing'] + test_df['StDev'], alpha=0.2)
    ax.set(xlabel='Number of Training Instances', ylabel=ylabel, title=save_name)

    ratio = 0.5
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.legend()
    ax.grid()

    #oname = '_'.join([clf.__class__.__name__, save_name]) + '.png'
    fig.savefig(save_name + '.png')

    return None

def plot_loss_curve(results):
    short_name = {'random_hill_climb': 'RHC', 'genetic_alg': 'GA', 'simulated_annealing': 'SA', 'backprop':'backprop'}
    for alg, val in results.items():
        fig, ax = plt.subplots()
        fig.set_figheight(3.5)
        algo = short_name[alg]
        # print(algo)
        # min_len = min([len(x) for x in val['fitness_curve']])
        min_len = min([len(x) for x in val])
        if alg == 'backprop':
            loss_curve = np.stack([c[:min_len] for c in val], axis=-1)
        else:
            loss_curve = np.stack([c[:min_len] for c in val], axis=-1)[:, 0]
        df_plot = pd.DataFrame([loss_curve.mean(axis=-1), loss_curve.std(axis=-1)], index=[algo, 'StdDev']).T
        df_plot[algo].plot(ax=ax, label=algo)
        ax.fill_between(df_plot.index, df_plot[algo]-df_plot['StdDev'], df_plot[algo]+df_plot['StdDev'], alpha=0.2)
        ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
        ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
        # ax.set(xlabel='Problem Size', ylabel='Wall Clock Time (s)', title=problem_name)
        title = '_'.join([alg, 'convergence'])
        ax.set(xlabel='Iteration', ylabel='Loss', title=title)
        ax.legend()
        ax.grid()
        fig.show()
        fig.savefig(title + '.png')

    return None


if __name__ == '__main__':

    if 'Assignment' not in os.getcwd():
        os.chdir('Assignment2')

    data_dir = './tic+tac+toe+endgame/'
    df = pd.read_csv(os.path.join(data_dir, 'tic-tac-toe.data'), header=None).sample(frac=1)
    X, y = df.iloc[:, :-1].replace(['x', 'o', 'b'], [2, 1, 0]), df.iloc[:, -1].replace(['negative', 'positive'],
                                                                                       [0, 1])
    os.chdir(data_dir)

    results = {}
    grid_params = dict(random_hill_climb=dict(restarts=[10],
                                              max_iters=[10000],
                                              ),
                       genetic_alg=dict(mutation_prob=[0.2, 0.5, 0.7],
                                        pop_size=[20, 50, 100, 200],
                                        max_iters=[1000]
                                        ),
                       simulated_annealing=dict(schedule=[mlrose.GeomDecay(1), mlrose.GeomDecay(50), mlrose.GeomDecay(500), mlrose.GeomDecay(5000), mlrose.GeomDecay(10000)],
                                                max_iters=[1000])
                       )

    # mutation_prob=0.5, pop_size=100
    results, do_grid_search = {}, False
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

    # for nn_algo, nn_algo_short in nn_algorithms.items():
    if do_grid_search:
        for nn_algo, param_grid in grid_params.items():

            print(nn_algo)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=[10], activation='relu',
                                            algorithm=nn_algo, max_iters=1000,
                                            bias=True, is_classifier=True, learning_rate=0.001,
                                            early_stopping=True, clip_max=500, max_attempts=100,
                                            random_state=9, curve=True)

            grid_search = GridSearchCV(estimator=nn_model,
                                       param_grid=param_grid,
                                       scoring='f1',
                                       cv=StratifiedKFold(n_splits=2, shuffle=True),
                                       return_train_score=True,
                                       verbose=1)

            grid_search.fit(X_train, y_train)
            print(grid_search.best_score_)
            print(grid_search.best_params_)
            # results[nn_algo] = grid_search.best_estimator_

    optim_params = dict(random_hill_climb=dict(restarts=10,
                                               max_iters=1000,
                                               ),
                        genetic_alg=dict(mutation_prob=0.7,
                                         pop_size=20,
                                         max_iters=1000
                                         ),
                        simulated_annealing=dict(schedule=mlrose.GeomDecay(),
                                                 max_iters=1000),
                        backprop=dict()
                        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    results, short_name = {}, {'random_hill_climb':'RHC', 'genetic_alg':'GA', 'simulated_annealing':'SA'}
    for nn_algo, algo_params in optim_params.items():
        print(nn_algo)
        for seed in [9]:
            if nn_algo != 'backprop':
                nn_model = mlrose.NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm=nn_algo,
                                                bias=True, is_classifier=True, learning_rate=0.001,
                                                early_stopping=True, clip_max=500, max_attempts=100,
                                                random_state=seed, curve=True)
                nn_model.set_params(**algo_params)
            else:
                nn_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                                         early_stopping=True, random_state=seed,
                                         learning_rate_init=0.001)

            train_scores, test_scores, fit_times, score_times = plot_learning_curve(nn_model, X, y, nn_algo)

        fitness_curves = []
        for seed in [9, 10, 15]:
            if nn_algo != 'backprop':
                nn_model = mlrose.NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm=nn_algo,
                                                bias=True, is_classifier=True, learning_rate=0.001,
                                                early_stopping=True, clip_max=500, max_attempts=100,
                                                random_state=seed, curve=True)
                nn_model.set_params(**algo_params)
                nn_model.fit(X_train, y_train)
                fitness_curves.append(nn_model.fitness_curve)
            else:
                nn_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                                         early_stopping=True, random_state=seed,
                                         learning_rate_init=0.001)
                nn_model.fit(X_train, y_train)
                fitness_curves.append(nn_model.loss_curve_)

        results[nn_algo] = fitness_curves

    plot_loss_curve(results)