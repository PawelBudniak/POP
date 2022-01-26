import random
import math
import numpy as np

SIGMA_RANGE = 0.1
MEAN = 0
SIGMA = 1


def next_sigma(sigma, a, dimension):
    b = np.random.normal(MEAN, SIGMA)
    tau = 1 / math.sqrt(2 * dimension)
    tau_prim = 1 / math.sqrt(2 * math.sqrt(dimension))
    sigma_j = sigma * math.exp(tau_prim * a + tau * b)

    return sigma_j


class Genotype:

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None

        self.sigma = [random.uniform(1 - SIGMA_RANGE, 1 + SIGMA_RANGE)]

        a = np.random.normal(MEAN, SIGMA)
        dimension = len(self.chromosome)
        for i in range(1, dimension):
            sigma_j = next_sigma(self.sigma[i - 1], a, dimension)
            self.sigma.append(sigma_j)

    @classmethod
    def random_genotype(cls, bounds):
        chromosome = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
        return cls(chromosome)
