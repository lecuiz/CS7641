import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn import metrics
from scipy.spatial import distance
from time import time

def compute_bic(kmeans,X):
    # https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans/251169#251169
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X.iloc[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return BIC


def fit_and_evaluate(km, X, labels, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)

#    for n_clusters
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        pred_labels = km.labels_ if 'KMeans' in name else km.predict(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, pred_labels))
        scores["Completeness"].append(metrics.completeness_score(labels, pred_labels))
        scores["V-measure"].append(metrics.v_measure_score(labels, pred_labels))
        scores["Adjusted Rand-Index"].append(metrics.adjusted_rand_score(labels, pred_labels))
        scores["Silhouette Coefficient"].append(metrics.silhouette_score(X, pred_labels, sample_size=2000))
        if 'Gaussian' in name:
            scores["BIC"].append(km.bic(X))
        elif 'KMeans' in name:
            scores["BIC"].append(compute_bic(km, X))
    train_times = np.asarray(train_times)

    print(f"{name} clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        # "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        # "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        # print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score

    return evaluation, evaluation_std


def run_clusters(km, X, y):
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    cluster_list = range(2, 20)
    evals, eval_stds = [], []
    base_name = km.__name__
    for n_clusters in cluster_list:
        if 'KMeans' in base_name:
            km_ = km(n_clusters=n_clusters)
        elif 'Gaussian' in base_name:
            km_ = km(n_components=n_clusters)
        name = '_'.join([base_name, str(n_clusters)])
        eval, eval_std = fit_and_evaluate(km_, X, y, name=name, n_runs=5)
        evals.append(eval)
        eval_stds.append(eval_std)

    df_evals = pd.DataFrame(evals, index=cluster_list)
    df_std = pd.DataFrame(eval_stds, index=cluster_list)

    #if 'BIC' in df_evals.columns:
    #    # convert std to percent
    #    df_std['BIC'] = df_std['BIC'] / df_evals['BIC']
    #    # 0, 1 range
    #    max_score = df_evals.drop(columns=['BIC']).max().max()
    #    min_score = df_evals.drop(columns=['BIC']).min().min()
    #    df_evals['BIC'] = (df_evals['BIC']-df_evals['BIC'].min())/(df_evals['BIC'].max() -df_evals['BIC'].min())
    #    df_evals['BIC'] *= max_score
    #    df_std['BIC'] *= max_score

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    for key in df_evals.columns:
        if key not in ['train_time', 'BIC']:
            df_evals[key].plot(ax=ax, style='-o')
            ax.fill_between(df_std.index, df_evals[key] - df_std[key], df_evals[key] + df_std[key], alpha=0.2)
    # ax.set_ylim(-0.01,0.5)
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.set(xlabel='Number of Clusters', ylabel='Score', title=base_name)
    ax.legend()
    ax.grid()
    fig.show()

    # oname = '_'.join([problem_name, title])
    oname = '_'.join([base_name, 'n_clusters'])
    fig.savefig(oname + '.png')

    return df_evals, df_std


def run_em(X, y):
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    param_grid = {"n_components": range(1, 20),
                  "covariance_type": ["spherical", "tied", "diag", "full"]}
    grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
    grid_search.fit(X)

    df = pd.DataFrame(grid_search.cv_results_)[["param_n_components", "param_covariance_type", "mean_test_score"]]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(columns={"param_n_components": "Number of components",
                            "param_covariance_type": "Type of covariance",
                            "mean_test_score": "BIC score"})
    df.sort_values(by="BIC score").head()

    df.plot.bar()

    return None

def run_pca(X, y, title):
    pca = PCA(0.999)
    pca.fit(X)
    pca.explained_variance_ratio_
    df_var = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance'])
    df_var['Explained Variance Ratio'] = df_var['Explained Variance'].cumsum()
    df_var.index += 1

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    df_var['Explained Variance'].plot.bar(ax=ax)
    df_var['Explained Variance Ratio'].plot(ax=ax, style='-o')
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.set(xlabel='Principal Component', ylabel='Explained Variance', title=title)
    ax.legend()
    ax.grid()
    fig.show()

    oname = '_'.join([title, 'pca'])
    fig.savefig(oname + '.png')

    return None
def run_ica(X, y, dataset):

    components = range(1, 10)
    for n_components in components:
        ica = FastICA(n_components=n_components)
        temp = ica.fit_transform(X)
        temp = pd.DataFrame(temp).kurt(axis=0)
        print(temp)
        temp = temp.kurt(axis=0)
        # kurt.append(temp.abs().mean())

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    #ax.plot(dims, kurt, 'b-')
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    # title = "ICA Kurtosis: "+ dataset
    # ax.set(xlabel="Independent Components", ylabel="Avg Kurtosis Across IC", title=title)
    ax.grid(False)
    ax.legend()
    fig.show()
    oname = '_'.join(dataset, 'ica')
    fig.savefig(oname + '.png')

def run_random(X, y):
    random_proj = SparseRandomProjection()
    random_proj.fit(X)
    n_features = X.shape[1]
    components = range(1, n_features)
    for n_components in components:
        rca = SparseRandomProjection(n_components=n_components)
        rca.fit(X)

    return None

def run_manifold(X, y):

    return None


if __name__ == '__main__':

    np.random.seed(0)
    np.random.RandomState(0)

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
    # 1. RUN 2 clusters (KMeans, GaussiaMix)
    # 2. RUN 4 dim reduce (PCA, ICA, RCA, manifold)
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

    for km in [KMeans, GaussianMixture]:
        df_evals, df_std = run_clusters(km, X, y)



