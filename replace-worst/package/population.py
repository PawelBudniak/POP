import optproblems
import numpy as np

from package.genotype import *


class Population:
    MU = 20
    LAMBDA = 7 * MU
    TOTAL = MU + LAMBDA

    def __init__(self, members, function, bounds, dimension, bias, worst_individuals_replacer=None):
        self.members = members
        self.dimension = dimension
        self.generation = 1
        self.replacer = worst_individuals_replacer
        self.bounds = bounds
        self.function = function
        self.bias = bias

        self.n_replaced = []

    @classmethod
    def rand_population(cls, function, bounds, dimension, bias, worst_individuals_replacer=None):
        members = [Genotype.random_genotype(bounds) for _ in range(cls.MU)]

        population = cls(members, function, bounds, dimension, bias, worst_individuals_replacer)
        population.evaluate(members, clip=False)
        return population

    def evolution(self):
        selection = self.select(self.LAMBDA)
        offspring = self.mate(selection)
        offspring = self.mutate(offspring)
        offspring = self.evaluate(offspring, self.function)
        self.members = self.succeed(self.members, offspring, self.MU)
        self.generation += 1

    def select(self, how_many):
        tmp_generation = []

        for _ in range(how_many):
            tmp_generation.append(random.choice(self.members))

        return tmp_generation

    def mate(self, members):
        children_genotypes = []
        # average with random weight
        for _ in members:
            child_chromosome = []
            parent_1 = random.choice(members)
            parent_2 = random.choice(members)
            weight = random.uniform(MEAN, SIGMA)

            for i in range(self.dimension):
                child_chromosome.append(weight * parent_1.chromosome[i] + (1 - weight) * parent_2.chromosome[i])

            child_genotype = Genotype(child_chromosome)
            children_genotypes.append(child_genotype)

        return children_genotypes

    def mutate(self, genotypes):
        for genotype in genotypes:
            genotype.chromosome = [
                gene + sigma * np.random.normal(MEAN, SIGMA)
                for gene, sigma, bound in zip(genotype.chromosome, genotype.sigma, self.bounds)
            ]
        return genotypes

    def evaluate(self, genotypes, clip=True):
        if clip:
            for genotype in genotypes:
                genotype.chromosome = [np.clip(gene, bound[0], bound[1])
                                       for gene, bound in zip(genotype.chromosome, self.bounds)]

        for genotype in genotypes:
            individual = optproblems.base.Individual(genotype.chromosome)
            self.function.evaluate(individual)

            genotype.fitness = individual.objective_values - self.bias

        return genotypes

    def succeed(self, current_generation, children, population_size):
        next_generation = current_generation + children
        next_generation.sort(key=lambda x: x.fitness)

        prev_best = set(next_generation[:population_size])

        if self.replacer is not None:
            replaced = self.replacer.replace(next_generation, self.bounds)
            next_generation[-len(replaced):] = replaced
            next_generation = self.evaluate(next_generation)
            next_generation.sort(key=lambda x: x.fitness)
            new_best = set(next_generation[:population_size])
            self.n_replaced.append(len(prev_best) - len(prev_best.intersection(new_best)))
        else:
            self.n_replaced.append(0)

        return next_generation[:population_size]

    def avg_replaced(self):
        return sum(self.n_replaced) / (self.MU * len(self.n_replaced))
