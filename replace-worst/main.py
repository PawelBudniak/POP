from package.test import *
from sklearn.model_selection import ParameterGrid


def main():
    test_multiple_replacers()
    # test_single_param()
    # grid_search()


def test_multiple_replacers():
    n_worst = 0.2

    fun = 'gp'

    replacers = [None,
                 RandomReplacer(n_worst),
                 SampleAroundBestReplacer(n_worst, best_frac=0.05, stdev_scale=2.0),
                 PropagateExtremesReplacer(n_worst, best_frac=0.02, stdev_scale=1.0),
                 CompositeReplacer(n_worst)
                 ]
    x_axis = ['none', 'random', 'best-norm', 'prop-extremes', 'composite']

    test_output = run_tests(fun, replacers, iterations=40)
    show_test_output(test_output, params=x_axis, graph_filename=f'{fun}-testura', xlabel='Replacement method')


def test_single_param():
    n_worst = 0.2
    fun = 'f5'

    vals = [0.1, 0.5, 0.33, 0.05]
    x_axis = vals
    arg = 'n_best'
    replacers = [SampleAroundBestReplacer(n_worst, stdev_scale=1.0, **{arg: val}) for val in vals]

    test_output = run_tests(fun, replacers, iterations=10)
    show_test_output(test_output, params=x_axis, graph_filename=f'sample-best-{arg}', xlabel=arg)


def grid_search():
    fun = 'f6'

    ranges = {
        'worst_frac': [0.5, 0.33, 0.2, 0.1],
        'best_frac': [0.33, 0.1, 0.05, 0.02],
        'stdev_scale': [0.5, 1.0, 2.0, 4.0],
    }

    replacers_dict = {
        'random': RandomReplacer,
        'best_norm': SampleAroundBestReplacer,
        'prop_extremes': PropagateExtremesReplacer
    }

    for replacer_s, replacer in replacers_dict.items():
        if replacer_s == 'random':
            grid = {'worst_frac': ranges['worst_frac']}
        else:
            grid = ranges

        results = []
        for params in ParameterGrid(grid):
            replacers = [replacer(**params)]
            test_output = run_tests(fun, replacers, iterations=5)
            data, _ = test_output
            best_fit_overall, mean_best_fit, best_fit_standard_deviation, _ = data[0]

            results.append([*params.values(), best_fit_overall, mean_best_fit, best_fit_standard_deviation])

        fname = f'grid_search_{fun}_{replacer_s}.csv'
        with open(fname, 'wt') as fp:
            fp.write(','.join([*grid.keys(), 'best fit', 'mean best fit', 'best fit stdev']) + '\n')
            fp.writelines(','.join(map(str, result)) + '\n' for result in results)

        # log for colab
        with open(fname, 'rt') as fp:
            print(fp.read())


if __name__ == '__main__':
    main()
