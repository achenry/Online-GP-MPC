import numpy as np


def gradient_func(func, x, diff=1e-6):
    grad = [(func(np.hstack([x[:i], x[i] + diff, x[i + 1:]])) -
             func(np.hstack([x[:i], x[i] - diff, x[i + 1:]]))) / (2 * diff)
            for i in range(len(x))]

    return np.hstack(grad)


def jacobian_func(func, x, diff=1e-6):
    """
    if (np.abs(np.hstack([(func(np.hstack([x[:i], x[i] + diff, x[i + 1:]]))[np.newaxis, :].T -
                 func(np.hstack([x[:i], x[i] - diff, x[i + 1:]]))[np.newaxis, :].T) / (2 * diff)
                for i in np.arange(len(x))])) > 100).any():
        print(np.hstack([(func(np.hstack([x[:i], x[i] + diff, x[i + 1:]]))[np.newaxis, :].T -
          func(np.hstack([x[:i], x[i] - diff, x[i + 1:]]))[np.newaxis, :].T)
         for i in np.arange(len(x))]))
    """
    if func(x).ndim == 1:
        jac_cols = [(func(np.hstack([x[:i], x[i] + diff, x[i + 1:]]))[np.newaxis, :].T -
                     func(np.hstack([x[:i], x[i] - diff, x[i + 1:]]))[np.newaxis, :].T) / (2 * diff)
                    for i in range(len(x))]
    else:
        jac_cols = [(func(np.hstack([x[:i], x[i] + diff, x[i + 1:]])) -
                     func(np.hstack([x[:i], x[i] - diff, x[i + 1:]]))) / (2 * diff)
                    for i in range(len(x))]

    return np.hstack(jac_cols)
