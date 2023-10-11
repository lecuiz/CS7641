import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def plot_hyper(clf_base, hyper_kwarg, hyper_range, hyper_name, X, y):
    oname = '_'.join([clf_base.__name__, hyper_kwarg]) # + '.png'
    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    train_data, test_data = {}, {}
    for hyper in hyper_range:
        clf_kwargs = {hyper_kwarg: hyper}
        if hyper_kwarg == 'degree':
            clf_kwargs['kernel'] = 'poly'
        clf = clf_base(**clf_kwargs)
        print(clf_kwargs)
        cv = cross_validate(clf, X, y, return_train_score=True, cv=StratifiedKFold(n_splits=5, shuffle=True))
        test_data[hyper] = {'Testing': cv['test_score'].mean(), 'StDev': cv['test_score'].std()}
        train_data[hyper] = {'Training': cv['train_score'].mean(), 'StDev': cv['train_score'].std()}
    test_data = pd.DataFrame.from_dict(test_data, orient='index')
    train_data = pd.DataFrame.from_dict(train_data, orient='index')
    train_data.plot(ax=ax, style='-o', yerr='StDev')
    test_data.plot(ax=ax, style='-o', yerr='StDev')

    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    # tick_spacing = 0.05
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set(xlabel=hyper_name, ylabel='F1 Score', title=oname)
    ax.legend()
    ax.grid()

    fig.savefig(oname + '.png')
    return fig


def plot_learning_curve(clf, X, y, algo):

    # fig, ax = plt.subplots()
    # fig.set_figheight(4)
    train_sizes = np.linspace(0.1, 1.0, 30)
    lc = learning_curve(clf, X, y,
                        train_sizes=train_sizes, return_times=True,
                        cv=StratifiedKFold(n_splits=5, shuffle=True),
                        scoring='f1', shuffle=True)
    train_size, train_scores, test_scores, fit_times, score_times = lc

    oname = '_'.join([clf.__class__.__name__, 'learning_curve']) # + '.png'
    plot_from_df(train_scores, test_scores, train_size, 'F1 Score', oname)

    oname = '_'.join([clf.__class__.__name__, 'training_time']) # + '.png'
    plot_from_df(fit_times, score_times, train_size, 'Time (s)', oname)

    if algo == 'NN':
        oname = '_'.join([clf.__class__.__name__, 'epochs'])  # + '.png'
        plot_epochs(clf.loss_curve_, clf.validation_scores_, oname)

    return None

def plot_epochs(loss_curve, val_scores, save_name):

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    loss_df = pd.DataFrame(loss_curve, columns=['Training Loss'])
    val_df = pd.DataFrame(val_scores, columns=['Validation'])
    loss_df.plot(ax=ax, style='-')
    val_df.plot(ax=ax, style='-', color='orange')

#    ax2 = ax.twinx()

    ax.set(xlabel='Epochs', ylim=(None,1), ylabel='Loss', title=save_name)
#    ax2.set(ylabel='F1 Score')
    #ax2.legend()

    ratio = 0.5
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.legend()
    ax.grid()

    #oname = '_'.join([clf.__class__.__name__, save_name]) + '.png'
    fig.savefig(save_name + '.png')

    return None


def plot_from_df(train, test, train_size, ylabel, save_name):
    fig, ax = plt.subplots()
    fig.set_figheight(3.5)

    # Test/Train plot
    train_df = pd.DataFrame([train.mean(axis=1), train.std(axis=1)],
                            index=['Training', 'StDev'], columns=train_size).T
    test_df = pd.DataFrame([test.mean(axis=1), test.std(axis=1)],
                           index=['Testing', 'StDev'], columns=train_size).T

    train_df.plot(ax=ax, style='-o', yerr='StDev')
    test_df.plot(ax=ax, style='-o', yerr='StDev')
    ax.set(xlabel='Number of Training Instances', ylabel=ylabel, title=save_name)

    ratio = 0.5
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.legend()
    ax.grid()

    #oname = '_'.join([clf.__class__.__name__, save_name]) + '.png'
    fig.savefig(save_name + '.png')

    return None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    np.random.seed(0)
    np.random.RandomState(0)

    # X, y = make_classification(n_samples=100000, n_features=20, n_redundant=2, n_classes=2)
    # data_dir = '/Users/dermacintosh/Desktop/OMSCS/Fall2023/CS7641'
    if 'Assignment1' not in os.getcwd():
        os.chdir('Assignment1')

    data_set = sys.argv[1]

    if data_set == 'tic':
        data_dir = './tic+tac+toe+endgame/'
        df = pd.read_csv(os.path.join(data_dir,'tic-tac-toe.data'), header=None).sample(frac=1)
        X, y = df.iloc[:, :-1].replace(['x', 'o', 'b'], [2, 1, 0]), df.iloc[:, -1].replace(['negative', 'positive'],
                                                                                           [0, 1])
        os.chdir(data_dir)
    elif data_set == 'diag':
        data_dir = './breast+cancer+wisconsin+diagnostic/'
        df = pd.read_csv(os.path.join(data_dir,'wdbc.data'), header=None).sample(frac=1)
        X, y = df.iloc[:, 2:], df.iloc[:, 1].replace(['M', 'B'], [1, 0])
        os.chdir(data_dir)
    else:
        raise ValueError('Incorrect dataset specified, please run python main_ro.py tic or python main_ro.py diag')

    run_grid = {'DT':
                    {'clf':
                         DecisionTreeClassifier,
                     'param_grid': {
                         'max_depth': range(1, 25, 2),
                         'ccp_alpha': np.linspace(0, 0.025, 10)
                     },
                     'plot_name': {
                         'max_depth': 'Tree Depth',
                         'ccp_alpha': 'Pruning Strength'
                     }},
                'KNN':
                    {'clf':
                         KNeighborsClassifier,
                     'param_grid': {
                         'n_neighbors': [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 200],
                         'metric': ['manhattan', 'euclidean', 'cosine']
                     },
                     'plot_name': {
                         'n_neighbors': 'Number of Neighbours',
                         'metric': 'Distance Metric'
                     }},
                'Boosted':
                    {'clf':
                         AdaBoostClassifier,
                     'param_grid': {
                         'learning_rate': np.linspace(0.75, 2.25, 11),
                         'n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200]
                     },
                     'plot_name': {
                         'learning_rate': 'Learning Rate',
                         'n_estimators': 'Number of Estimators'
                     }},
                'SVM':
                    {'clf':
                         SVC,
                     'param_grid': {
                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'degree': range(1,10)
                     },
                     'plot_name': {
                         'kernel': 'Kernel Type',
                         'degree': 'Degree'
                     }},
                'NN':
                    {'clf':
                         MLPClassifier,
                     'param_grid': {
                         'hidden_layer_sizes': [1, 2, 5, 10, 25, 50, 100, 250, 500],
                         'activation': ['identity', 'logistic', 'relu']
                     },
                     'plot_name': {
                         'hidden_layer_sizes': 'Hidden Layer Size',
                         'activation': 'Activation Function'
                     }},
                }

    df_results, dict_params = pd.DataFrame(), {}

    for algo in run_grid.keys():

        params = run_grid[algo]
        clf, param_grid, plot_name = params['clf'], params['param_grid'], params['plot_name']

        for param_name, param_range in param_grid.items():
            print(algo, param_name)
            param_title = plot_name[param_name]
            plot_hyper(clf, param_name, param_range, param_title, X, y)

        #param_search = RandomizedSearchCV(estimator=clf(), param_distributions=param_grid, scoring='f1', cv=StratifiedKFold(n_splits=5, shuffle=True))
        # print(algo, 'starting gridsearch')

        estimator = clf(early_stopping=True, n_iter_no_change=100) if algo == 'NN' else clf()
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   scoring='f1',
                                   cv=StratifiedKFold(n_splits=5, shuffle=True),
                                   return_train_score=True,
                                   verbose=1)
        grid_search.fit(X, y)

        clf_optimized, best_index = grid_search.best_estimator_, grid_search.best_index_
        print(algo, clf_optimized)
        plot_learning_curve(clf_optimized, X, y, algo)

        cols = ['params', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score',
                'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
        # train_score, test_score = cross_val['train_score'].mean(), cross_val['test_score'].mean()
        # train_time, test_time = cross_val['train_score'].mean(), cross_val['test_score'].mean()
        df_results[clf.__name__] = [grid_search.cv_results_[col][best_index] for col in cols]
        # dict_params[clf.__name__] = grid_search.cv_results_['params'][best_index]
        print(clf_optimized.get_params())

    df_results.index = cols
    df_results.T.to_csv('model_comparison.csv')

    df_results_ = df_results.drop(index='params').T
    df_results_ = df_results_.astype(float).round(3)
    df_results_.index.name = 'Algorithm'

    print_df_results = pd.DataFrame(index=df_results_.index)
    print_df_results['Train Score'] = df_results_['mean_train_score'].astype(str) + ' ± ' + df_results_['std_train_score'].astype(str)
    print_df_results['Test Score'] = df_results_['mean_test_score'].astype(str) + ' ± ' + df_results_['std_test_score'].astype(str)
    print_df_results['Training Time'] = df_results_['mean_fit_time'].astype(str) + ' ± ' + df_results_['std_fit_time'].astype(str)
    print_df_results['Testing Time'] = df_results_['mean_score_time'].astype(str) + ' ± ' + df_results_['std_score_time'].astype(str)
    print_df_results.to_csv('results.csv')

    # dict_params.index = ['params']
    # dict_params.T.to_csv('model_params.csv')


