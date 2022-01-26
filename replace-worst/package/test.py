from optproblems.cec2005 import F5, F6
from optproblems.continuous import Branin, GoldsteinPrice

from package.optimize import *
from package.data_storing import *
from package.visuals import *

from statistics import mean
from package.replace_worst import *

F5_BOUND = 100
F6_BOUND = 100
F5_OPT_BIAS = -310
F6_OPT_BIAS = 390

PARAMS = 4


def show_test_output(test_output, params, graph_filename, xlabel):
    data, boxplot_data = test_output
    output_data = FunctionOptimizationData(data, params)
    plot_boxplot(boxplot_data, params, graph_filename, xlabel)
    output_data.print_stats()


def run_tests(function, replacers, iterations):
    data = []
    boxplot_data = []

    if function == 'branin':  # Branin’s test problem ‘RCOS’
        f = Branin()
        # he search space is [-5, 0] \times [10, 15]. Every optimum is a global optimum.
        dimension = 2
        bound = [(-5, 0), (10, 15)]
        bias = 0

    if function == 'gp':  # Goldstein-Price function
        f = GoldsteinPrice()
        # The search space is [-2, 2] \times [-2, 2]
        dimension = 2
        bound = symmetric_bounds(bound=2, dim=dimension)
        bias = 0
    elif function == 'f5':  # Schwefel’s Problem 2.6 with Global Optimum on Bounds
        bias = F5_OPT_BIAS
        dimension = 10
        f = F5(dimension)
        bound = symmetric_bounds(F5_BOUND, dimension)
    elif function == 'f6':  # Shifted Rosenbrock’s Function
        bias = F6_OPT_BIAS
        dimension = 10
        f = F6(dimension)
        bound = symmetric_bounds(F6_BOUND, dimension)

    for replacer in replacers:
        print('Running {} by {} - No runs: {}...'.format(function, replacer, iterations))

        # runs = [optimize(f, bound, replacer, dimension, bias) for _ in range(iterations)]

        runs = []
        avg_replaceds = []
        for _ in range(iterations):
            best_evals, number_of_evals, avg_replaced = optimize(f, bound, replacer, dimension, bias)
            runs.append([best_evals, number_of_evals])
            avg_replaceds.append(avg_replaced)

        print(f'Replacer: {replacer}, avg replaced: {mean(avg_replaceds)}')

        data.append(merge_data(runs))
        boxplot_data.append(boxplot_from_multiple_runs(runs))

    return [data, boxplot_data]


def symmetric_bounds(bound, dim):
    return [(-bound, bound) for _ in range(dim)]
