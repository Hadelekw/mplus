import math
import numpy as np
from numbers import Number


def add(*args) -> Number:
    return max(args)


def mult(*args) -> Number:
    return sum(args) if -math.inf not in args else -math.inf


def add_matrices(A : np.ndarray,
                 B : np.ndarray) -> np.ndarray:
    if A.shape != B.shape:
        raise ValueError(
            'Maxplus.add_matrices: given matrices ' +\
            'are of different shape (A: {}, B: {})'.format(A.shape, B.shape)
        )
    result = np.copy(A)
    shape = A.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i, j] = add(result[i, j], B[i, j])
    return result


def mult_matrices(A : np.ndarray,
                  B : np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            'Maxplus.mult_matrices: given matrices ' +\
            'are of shapes not given as MxN and NxP (A: {}, B: {})'.format(
                A.shape, B.shape
            )
        )
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = add(*[mult(A[i, k], B[k, j]) for k in range(A.shape[1])])
    return result
