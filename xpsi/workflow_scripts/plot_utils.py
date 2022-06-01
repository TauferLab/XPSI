import numpy as np


metric_functions = {
    'Maximum': (lambda err: np.max(err, axis=1)),
    'Median': (lambda err: np.median(err, axis=1)),
    'Root Mean Squared': (lambda err: np.sqrt(np.mean(np.square(err), axis=1))),
    'Mean': (lambda err: np.mean(err, axis=1))
}
def generate_statistics(dataset_files, metrics):
    values = {name:[] for name in metrics}
    for dataset in dataset_files:
        config_errors = np.loadtxt(dataset)
        for m in metrics:
            values[m].append(metric_functions[m](config_errors))
    return values