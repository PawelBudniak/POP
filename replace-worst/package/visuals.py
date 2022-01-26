import matplotlib.pyplot as plt
import package.test as test

BEST_EVALS_IDX = 0


def boxplot_from_multiple_runs(runs):
    data = []

    for i in range(len(runs)):
        number_of_generations = len(runs[i][BEST_EVALS_IDX])
        # add best fit from run to data
        data.append(runs[i][BEST_EVALS_IDX][number_of_generations - 1])

    return data


def plot_boxplot(data, params, graph_filename, xlabel):
    fig, ax = plt.subplots()

    if params[3] == 715:
        params[3] = 'budget/lambda'

    ax.set_title(f'Best fits')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Best fit')

    ax.boxplot(data, labels=params)

    plt.savefig('graphs/' + graph_filename + '.png')
