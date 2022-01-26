from package.properties import *
import math

DECIMAL_POINTS = 2

BEST_FITS_IDX = 0
EVALS_IDX = 1
GENERATION_ZERO_SHIFT = 1

K_ITER_DATA_IDX = 0
SD_DATA_IDX = 1
BEST_WORST_DATA_IDX = 2
VARIANCE_DATA_IDX = 3


def merge_data(multiple_runs):  # merge [[run1_data], [run2_data], ...] into [mean_run_data]

    number_of_runs = len(multiple_runs)
    best_fit_overall = multiple_runs[0][BEST_FITS_IDX][0]
    best_fits_each_run = []
    mean_best_fit = 0
    mean_evals = 0

    for i in range(number_of_runs):
        number_of_generations = len(multiple_runs[i][BEST_FITS_IDX])
        best_fit_in_run = multiple_runs[i][BEST_FITS_IDX][number_of_generations - GENERATION_ZERO_SHIFT]
        best_fits_each_run.append(best_fit_in_run)

        if best_fit_in_run <= best_fit_overall:
            best_fit_overall = best_fit_in_run

        mean_best_fit += best_fit_in_run
        mean_evals += multiple_runs[i][EVALS_IDX]

    mean_best_fit /= number_of_runs
    mean_evals /= number_of_runs

    best_fit_standard_deviation = calc_list_sd(best_fits_each_run, mean_best_fit)

    return [best_fit_overall, mean_best_fit, best_fit_standard_deviation, mean_evals]


class FunctionOptimizationData:

    def __init__(self, data, params):
        self.crit_data = data
        self.params = params
        self.number_of_params = len(params)

    def print_stats(self):
        print('Param. value\t|\tBest fit\t|\tBest fit mean\t|\tBest fit standard deviation\t|\tNumber of evals mean')
        for i in range(self.number_of_params):
            print("{}\t\t|\t{} \t|\t{} \t|\t{} \t|\t{}"
                  .format(self.params[i],
                          round(self.crit_data[i][0], DECIMAL_POINTS),
                          round(self.crit_data[i][1], DECIMAL_POINTS),
                          round(self.crit_data[i][2], DECIMAL_POINTS),
                          round(self.crit_data[i][3], DECIMAL_POINTS)))
