import numpy as np
from scipy import stats


def upper_to_full(a, eps=0):
    if eps is not None:
        mask = (a < eps) & (a > -eps)
        a[mask] = 0
    n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A

