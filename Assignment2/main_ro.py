import os
import mlrose_hiive as mlrose
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold
import pandas as pd
import numpy as np

def plot_iterations(problem_results, problem_name):
    # create fit plot
    # plots = {'iterations': 'fit_curve', 'fevals': 'feval_curve'}
    # for title, curve_name in plots.items():
    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    # df_dict = {k: pd.DataFrame(val[curve_name], columns=[k, 'StdDev']) for k, val in problem_results.items()}
    df_dict = {k: pd.DataFrame(val['fit_curve'], columns=[k, 'StdDev']) for k, val in problem_results.items()}
    for key, df in df_dict.items():
        # df.plot(ax=ax, yerr='StdDev', label=key)
        df[key].plot(ax=ax, label=key)
        ax.fill_between(df.index, df[key]-df['StdDev'], df[key]+df['StdDev'], alpha=0.2)
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    # ax.set_xscale('log')
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.set(xlabel='Iteration', ylabel='Fitness', title=problem_name)
    ax.legend()
    ax.grid()
    fig.show()
    # oname = '_'.join([problem_name, title])
    oname = '_'.join([problem_name, 'iterations'])
    fig.savefig(oname + '.png')

    return None

def plot_problem_size(results_size, problem_name):

    plots = {'fitness': 'Fitness', 'runtime': 'Wall Clock Time (s)', 'fevals': 'Function Evaluations'}
    for plotname, ylab in plots.items():
        df_plots = {k: v[plotname] for k, v in results_size.items()}

        fig, ax = plt.subplots()
        fig.set_figheight(3.5)
        for k, df_plot in df_plots.items():
            # df_plot.plot(ax=ax, yerr='StdDev', label=key)
            df_plot[k].plot(ax=ax, style='-o', label=k)
            ax.fill_between(df_plot.index, df_plot[k]-df_plot['StdDev'], df_plot[k]+df_plot['StdDev'], alpha=0.2)
        ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
        ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
        # ax.set(xlabel='Problem Size', ylabel='Wall Clock Time (s)', title=problem_name)
        title = '_'.join([problem_name, plotname])
        ax.set(xlabel='Problem Size', ylabel=ylab, title=title)
        ax.legend()
        ax.grid()
        fig.show()
        fig.savefig(title + '.png')

    # fit_xy, time_xy = [], []
    # for key, val in results_size.items():
    #     xy = [val_['best_fitness'] for _, val_ in val.items()]
    #     fit_xy.append(xy)
    #     xy = [val_['run_time'].seconds for _, val_ in val.items()]
    #     time_xy.append(xy)
    # df_fit = pd.DataFrame(fit_xy, index=results_size.keys(), columns=val.keys())
    # df_time = pd.DataFrame(time_xy, index=results_size.keys(), columns=val.keys())

    return None

def run_all_algorithms(algo_dict, problem, return_all=True):
    problem_results = {}
    general_params = dict(max_attempts=100, curve=True)
    for algo_name, algo in algo_dict.items():
        print(key, algo_name)
        algo_function, algo_params = algo['fn'], algo['params']
        fitness, curves, times = [], [], []
        for seed in [10, 13, 16, 19]:
            general_params['random_state'] = seed
            start = datetime.now()
            best_state, best_fitness, fitness_curve = algo_function(problem=problem, **general_params, **algo_params)
            run_time = datetime.now() - start
            fitness.append(best_fitness)
            curves.append(fitness_curve)
            times.append(run_time)

        min_len = min([len(x) for x in curves])
        curves = np.stack([c[:min_len] for c in curves], axis=-1)
        mean_curve, std_curve = np.mean(curves, axis=-1), np.std(curves, axis=-1 )
        fit_curve = np.stack([mean_curve[:,0], std_curve[:,0]], axis=-1)
        # feval_curve = np.stack([mean_curve[:,1], std_curve[:,1]], axis=-1)

        if return_all:
            problem_results[algo_name] = dict(best_state=best_state, best_fitness=np.mean(fitness), fit_curve=fit_curve,
                                              run_time=np.mean(times))
        else:
            problem_results[algo_name] = dict(best_fitness=fitness, run_time=times)

    return problem_results

def run_problem_sizes(algo, algo_name, fitness_fn):
    general_params = dict(max_attempts=100, curve=True)
    algo_function, algo_params = algo['fn'], algo['params']
    problem_sizes = [10, 25, 50, 75, 100, 200]
    # df_fitness = pd.DataFrame(index=problem_sizes, columns=[algo_name, 'StDev'])
    # df_runtime = pd.DataFrame(index=problem_sizes, columns=[algo_name, 'StDev'])
    # df_devals =  pd.DataFrame(index=problem_sizes, columns=[algo_name, 'StDev'])
    fitness_list, runtime_list, feval_list, feval_time_list = [], [], [], []
    for problem_size in problem_sizes:
        print(key, algo_name, problem_size)
        problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness_fn, maximize=True)
        problem.set_mimic_fast_mode(True)
        # results_size[problem_size] = run_single_algorithm(algo, problem)
        fitness, curves, times, feval_time = [], [], [], []
        for seed in [10, 13, 16, 19]:
            general_params['random_state'] = seed
            start = datetime.now()
            _, best_fitness, fitness_curve = algo_function(problem=problem, **general_params, **algo_params)
            run_time = datetime.now() - start
            fitness.append(best_fitness)
            curves.append(fitness_curve)
            times.append(run_time.total_seconds())
            feval_time.append(fitness_curve[-1,1]/run_time.total_seconds())

        fitness_list.append([np.mean(fitness), np.std(fitness)])
        runtime_list.append([np.mean(times), np.std(times)])
        fevals = [x[-1,1] for x in curves]
        feval_list.append([np.mean(fevals), np.std(fevals)])
        feval_time_list.append([np.mean(feval_time), np.std(feval_time)])

    df_fitness = pd.DataFrame(fitness_list, index=problem_sizes, columns=[algo_name, 'StdDev'])
    df_runtime = pd.DataFrame(runtime_list, index=problem_sizes, columns=[algo_name, 'StdDev'])
    df_fevals = pd.DataFrame(feval_list, index=problem_sizes, columns=[algo_name, 'StdDev'])
    df_feval_per_time = pd.DataFrame(feval_time_list, index=problem_sizes, columns=[algo_name, 'StdDev'])

    return dict(fitness=df_fitness, runtime=df_runtime, fevals=df_fevals, feval_per_time=df_feval_per_time)


if __name__ == '__main__':

    general_params = dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
    run_dict = dict(
        FlipFlop=dict(fitness_fn=mlrose.FlipFlop(),
                      algos=dict(RHC=dict(fn=mlrose.random_hill_climb, params=dict(restarts=25, max_iters=1000)),
                                 SA=dict(fn=mlrose.simulated_annealing, params=dict(max_iters=1000, schedule=mlrose.GeomDecay(init_temp=25))),
                                 GA=dict(fn=mlrose.genetic_alg, params=dict(pop_size=100, mutation_prob=0.7, max_iters=100)),
                                 MIMIC=dict(fn=mlrose.mimic, params=dict(pop_size=200, keep_pct=0.5, max_iters=40))
                                 )),
        OneMax=dict(fitness_fn=mlrose.OneMax(),
                    algos=dict(RHC=dict(fn=mlrose.random_hill_climb, params=dict(restarts=25, max_iters=1000)),
                               SA=dict(fn=mlrose.simulated_annealing, params=dict(max_iters=1000)),
                               GA=dict(fn=mlrose.genetic_alg, params=dict(pop_size=50, mutation_prob=0.2, max_iters=100)),
                               MIMIC=dict(fn=mlrose.mimic, params=dict(pop_size=200, keep_pct=0.25, max_iters=40))
                               )),
        FourPeaks=dict(fitness_fn=mlrose.FourPeaks(),
                       algos=dict(RHC=dict(fn=mlrose.random_hill_climb, params=dict(restarts=25, max_iters=1000)),
                                  SA=dict(fn=mlrose.simulated_annealing, params=dict(max_iters=1000)),
                                  GA=dict(fn=mlrose.genetic_alg, params=dict(pop_size=200, mutation_prob=0.7, max_iters=100)),
                                  MIMIC=dict(fn=mlrose.mimic, params=dict(pop_size=100, keep_pct=0.25, max_iters=40))
                                  ))
    )

                    # # problem=mlrose.DiscreteOpt(length=100, fitness_fn=mlrose.FourPeaks(), maximize=True),
                    #      # algos=dict(RHC=mlrose.random_hill_climb)
                    # KnapSack=dict(fitness_fn=mlrose.Knapsack(),
                    #               #params=dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
                    #               ),
                    # TravellingSales=dict(fitness_fn=mlrose.TravellingSales(),
                    #                      # params=dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
                    #               )
                    # )

    if 'Assignment' not in os.getcwd():
        os.chdir('Assignment2')

    results = {}
    for key, val in run_dict.items():
        # fitness_fn, params = val['fitness_fn'], val['params']
        fitness_fn, algo_dict = val['fitness_fn'], val['algos']
        problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True)
        problem.set_mimic_fast_mode(True)
        # problem_results = {}
        # for algo_name, algo in algo_dict.items():
        #     print(key, algo_name)
        #     algo_function, algo_params = algo['fn'], algo['params']
        #     start = datetime.now()
        #     best_state, best_fitness, fitness_curve = algo_function(problem=problem, **general_params, **algo_params)
        #     run_time = datetime.now() - start
        #     problem_results[algo_name] = dict(best_state=best_state, best_fitness=best_fitness, fit_curve=fitness_curve, run_time=run_time)
        # results[key] = problem_results
        results[key] = run_all_algorithms(algo_dict, problem)
        plot_iterations(results[key], key)

        results_size = {}
        for algo_name, algo in algo_dict.items():
            results_size[algo_name] = run_problem_sizes(algo, algo_name, fitness_fn)

        plot_problem_size(results_size, key)
