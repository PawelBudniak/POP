from package.population import *
from package.properties import *


def optimize(function, bounds, replacer, dimension, bias):
    population = Population.rand_population(function, bounds, dimension, bias, replacer)

    best_evals = []
    number_of_evals = Population.MU  # MU is the number of P0 evals

    if dimension == 10:
        budget = 10000 * dimension
    else:
        budget = 1000 * dimension

    n_evals = []
    while number_of_evals + Population.LAMBDA < budget:
        n_evals.append(number_of_evals)
        best_evals.append(population.members[BEST_MEMBER].fitness)

        population.evolution()

        number_of_evals += Population.LAMBDA

    avg_replaced = population.avg_replaced()


    # plot_fitness_by_number_of_evals(best_evals, n_evals, function)
    return [best_evals, number_of_evals, avg_replaced]


def plot_fitness_by_number_of_evals(fitness, n_evals, function):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(n_evals, fitness)

    ax.set_title(f'{function.__class__.__name__} function fitness by number of evals')
    ax.set_xlabel('Number of evals')
    ax.set_ylabel('Best fit')

    ax.set_yscale("log")
    plt.show()

    exit(1)
