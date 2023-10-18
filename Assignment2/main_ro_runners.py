import numpy as np
import os
import pandas as pd
import mlrose_hiive as mlrose
from mlrose_hiive import GARunner, RHCRunner, SARunner, MIMICRunner
import matplotlib.pyplot as plt

def plot_GA_details(experiment_name, output_directory):
    fname = 'ga__' + experiment_name + '__curves_df.csv'
    fp = os.path.join(output_directory, experiment_name, fname)
    df = pd.read_csv(fp, index_col=0)

    for name, df_group in df.groupby(['Mutation Rate']):
        fig, ax = plt.subplots()
        fig.set_figheight(3.5)
        df_group.pivot(columns='Population Size', values='Fitness', index='Iteration').plot(ax=ax)
        ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
        # ax.set_xscale('log')
        title = experiment_name + ' Mutation Rate:' + str(name)
        ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
        ax.set(xlabel='Iteration', ylabel='Fitness', title=title)
        ax.legend()
        ax.grid()
        fig.show()
        oname = '_'.join([experiment_name, 'mutation_rate', str(name)])

        fig.savefig(oname + '.png')

def plot_SA_details(experiment_name, output_directory):
    fname = 'sa__' + experiment_name + '__curves_df.csv'
    fp = os.path.join(output_directory, experiment_name, fname)
    df = pd.read_csv(fp, index_col=0)

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    for name, df_group in df.groupby(['Temperature']):
        df_group.plot(x='Iteration', y='Fitness', ax=ax, label=name)
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    # ax.set_xscale('log')
    title = ' '.join([experiment_name, 'Temperature'])
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.set(xlabel='Iteration', ylabel='Fitness', title=title)
    ax.legend()
    ax.grid()
    fig.show()

    oname = '_'.join([experiment_name, 'temperature'])
    fig.savefig(oname + '.png')

    fig, ax = plt.subplots()
    fig.set_figheight(3.5)
    for name, df_group in df.groupby(['Temperature']):
        df_group.plot(x='Time', y='FEvals', ax=ax, label=name)
    ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
    # ax.set_xscale('log')
    title = ' '.join([experiment_name, 'FEvals'])
    ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
    ax.set(xlabel='Wall Clock Time (s)', ylabel='Funcion Evaluations', title=title)
    ax.legend()
    ax.grid()
    fig.show()

    oname = '_'.join([experiment_name, 'fevals'])
    fig.savefig(oname + '.png')

    return None

def plot_MIMIC_details(experiment_name, output_directory):
    fname = 'mimic__' + experiment_name + '__curves_df.csv'
    fp = os.path.join(output_directory, experiment_name, fname)
    df = pd.read_csv(fp, index_col=0)

    for name, df_group in df.groupby(['Keep Percent']):
        fig, ax = plt.subplots()
        fig.set_figheight(3.5)
        df_group.pivot(columns='Population Size', values='Fitness', index='Iteration').plot(ax=ax)
        ratio, xlim, ylim = 0.5, ax.get_xlim(), ax.get_ylim()
        # ax.set_xscale('log')
        title = experiment_name + ' Keep Percent:' + str(name)
        ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[0] - ylim[1])) * ratio)
        ax.set(xlabel='Iteration', ylabel='Fitness', title=title)
        ax.legend()
        ax.grid()
        fig.show()
        oname = '_'.join([experiment_name, 'keep_pct', str(name)])
        fig.savefig(oname + '.png')

    return None


general_params = dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
run_dict = dict(
                FlipFlop=dict(fitness_fn=mlrose.FlipFlop(),
                              # params=dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
                              ),
                OneMax=dict(fitness_fn=mlrose.OneMax(),
                            # params=dict(max_attempts=100, max_iters=100, random_state=13, curve=True)
                            ),
                FourPeaks=dict(fitness_fn=mlrose.FourPeaks(),
                               )
)

if 'Assignment' not in os.getcwd():
    os.chdir('Assignment2')

for key, val in run_dict.items():
    # fitness_fn, params = val['fitness_fn'], val['params']
    fitness_fn = val['fitness_fn']
    if key == 'FourPeaks':
        problem = mlrose.DiscreteOpt(length=50, fitness_fn=fitness_fn, maximize=True)
    else:
        problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True)
    problem.set_mimic_fast_mode(True)

    # experiment_name, output_directory = '_'.join([key + 'RHC']), key
    # print('Optimising ', experiment_name)
    # rhc = RHCRunner(problem=problem,
    #                 experiment_name=experiment_name,
    #                 output_directory=output_directory,
    #                 seed=13,
    #                 iteration_list=[1000],
    #                 max_attempts=100,
    #                 restart_list=[10, 25, 50])
    # df_run_stats, df_run_curves = rhc.run()
    # print(df_run_stats.sort_values(by='Fitness').tail(5))

    if key == 'FlipFlop':
        experiment_name, output_directory = '_'.join([key + 'SA']), key
        print('Optimising ', experiment_name)
        sa = SARunner(problem=problem,
                       experiment_name=experiment_name,
                       output_directory=output_directory,
                       seed=13,
                       # iteration_list=[100],
                       iteration_list=[1000],
                       max_attempts=100,
                       temperature_list=[1, 10, 25, 50, 100, 250, 500, 1000, 2500],
                       decay_list=[mlrose.GeomDecay])
        df_run_stats, df_run_curves = sa.run()
        print(df_run_stats.sort_values(by='Fitness').tail(5))
        plot_SA_details(experiment_name, output_directory)

    if key == 'FourPeaks':
        experiment_name, output_directory = '_'.join([key + 'GA']), key
        print('Optimising ', experiment_name)
        ga = GARunner(problem=problem,
                      experiment_name=experiment_name,
                      output_directory=output_directory,
                      seed=13,
                      iteration_list=[100],
                      max_attempts=100,
                      population_sizes=[20, 50, 100, 200],
                      mutation_rates=[0.2, 0.5, 0.7])
        df_run_stats, df_run_curves = ga.run()
        print(experiment_name, df_run_stats.sort_values(by='Fitness').tail(5))
        plot_GA_details(experiment_name, output_directory)

    if key == 'OneMax':
        experiment_name, output_directory = '_'.join([key + 'MIMIC']), key
        print('Optimising ', experiment_name)
        mmc = MIMICRunner(problem=problem,
                          experiment_name=experiment_name,
                          output_directory=output_directory,
                          use_fast_mimic=True,
                          seed=13,
                          iteration_list=[40],
                          max_attempts=100,
                          population_sizes=[20, 50, 100, 200],
                          keep_percent_list=[0.25, 0.5, 0.75])
        df_run_stats, df_run_curves = mmc.run()
        print(experiment_name, df_run_stats.sort_values(by='Fitness').tail(5))
        plot_MIMIC_details(experiment_name, output_directory)

