import numpy as np
from package.genotype import Genotype
import abc


class Replacer(abc.ABC):
    def __init__(self, worst_frac):
        self.worst_frac = worst_frac

    @abc.abstractmethod
    def replace(self, sorted_population, bounds):
        pass

    @staticmethod
    def _frac_to_flat(pop, frac):
        return int(frac * len(pop))


class RandomReplacer(Replacer):
    def __init__(self, worst_frac):
        super().__init__(worst_frac)

    def replace(self, sorted_population, bounds):
        n_worst = self._frac_to_flat(sorted_population, self.worst_frac)
        replaced = [Genotype.random_genotype(bounds) for _ in range(n_worst)]

        return replaced


class SampleAroundBestReplacer(Replacer):

    def __init__(self, worst_frac, best_frac, stdev_scale, dist=np.random.normal):
        super().__init__(worst_frac)
        self.dist = dist
        self.stdev_scale = stdev_scale
        self.best_frac = best_frac

    def replace(self, sorted_population, bounds):
        n_best = self._frac_to_flat(sorted_population, self.best_frac)
        n_worst = self._frac_to_flat(sorted_population, self.worst_frac)

        best_genotypes = sorted_population[:n_best]

        gene_matrix = np.array([genotype.chromosome for genotype in best_genotypes])
        means = np.mean(gene_matrix, axis=0)
        stdevs = np.std(gene_matrix, axis=0)

        replaced = []
        for _ in range(n_worst):

            chromosome = []
            for gene_mean, gene_stdev in zip(means, stdevs):
                chromosome.append(self.dist(loc=gene_mean, scale=gene_stdev * self.stdev_scale))

            replaced.append(Genotype(chromosome))

        return replaced


class PropagateExtremesReplacer(Replacer):

    def __init__(self, worst_frac, best_frac, stdev_scale, dist=np.random.normal):
        super().__init__(worst_frac)
        self.dist = dist
        self.stdev_scale = stdev_scale
        self.best_frac = best_frac

    def replace(self, sorted_population, bounds):
        n_best = self._frac_to_flat(sorted_population, self.best_frac)
        n_worst = self._frac_to_flat(sorted_population, self.worst_frac)

        best_genotypes = sorted_population[:n_best]


        gene_matrix = np.array([genotype.chromosome for genotype in best_genotypes])
        maxes = np.max(gene_matrix, axis=0)
        mins = np.min(gene_matrix, axis=0)
        stdevs = np.std(gene_matrix, axis=0)

        replaced = []
        for _ in range(n_worst):

            chromosome = []
            for gene_max, gene_min, gene_stdev in zip(maxes, mins, stdevs):
                sample = self.dist(loc=0, scale=gene_stdev * self.stdev_scale)

                extreme = gene_max if sample > 0 else gene_min
                sample += extreme
                chromosome.append(sample)

            replaced.append(Genotype(chromosome))

        return replaced


class CompositeReplacer(Replacer):

    def __init__(self, worst_frac):
        super().__init__(worst_frac)
        self.replacers = [
            RandomReplacer(worst_frac=worst_frac * 0.1),
            SampleAroundBestReplacer(worst_frac=worst_frac * 0.45, best_frac=0.05, stdev_scale=2.0),
            PropagateExtremesReplacer(worst_frac=worst_frac * 0.45, best_frac=0.02, stdev_scale=1.0)
        ]

    def replace(self, sorted_population, bounds):
        replaced = [r.replace(sorted_population, bounds) for r in self.replacers]

        return self._flatten(replaced)

    @staticmethod
    def _flatten(nested_list):
        return [item for sublist in nested_list for item in sublist]

