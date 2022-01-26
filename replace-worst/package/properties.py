import math

BEST_MEMBER = 0


def calc_fit_ev(population):
    expected_value = 0
    population_size = len(population.members)

    for i in range(population_size):
        expected_value += population.members[i].fitness

    expected_value /= population_size

    return expected_value


def calc_list_sd(values, expected):
    length = len(values)
    tmp_sum = 0
    for i in range(length):
        tmp_sum += math.pow(values[i] - expected, 2)

    standard_deviation = math.sqrt(tmp_sum / length)

    return standard_deviation


def calc_variance(population):
    expected_fit = calc_fit_ev(population)
    population_size = len(population.members)

    tmp_sum = 0
    for i in range(population_size):
        tmp_sum += math.pow(population.members[i].fitness - expected_fit, 2)

    fitness_variance = 1 / population_size * tmp_sum

    return fitness_variance
