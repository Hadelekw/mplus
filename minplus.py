import math
import numpy as np
from numbers import Number


def add(*args) -> Number:
    return min(args)


def mult(*args) -> Number:
    return sum(args) if math.inf not in args else math.inf


def add_matrices(A : np.ndarray,
                 B : np.ndarray) -> np.ndarray:
    if A.shape != B.shape:
        raise ValueError(
            'Minplus.add_matrices: given matrices ' +\
            'are of different shape (A: {}, B: {}).'.format(A.shape, B.shape)
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
            'Minplus.mult_matrices: given matrices ' +\
            'are of shapes not given as MxN and NxP (A: {}, B: {}).'.format(
                A.shape, B.shape
            )
        )
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = add(*[mult(A[i, k], B[k, j]) for k in range(A.shape[1])])
    return result


def modulo(a : Number, t : int) -> Number:
    if a == math.inf:
        return math.inf
    if a == 0:
        return 0
    if t == math.inf or t == 0:
        return a
    return a - (a // t) * t


def modulo_matrices(A : np.ndarray,
                    b : np.ndarray) -> np.ndarray:
    if b.shape[1] != 1:
        raise ValueError(
            'Minplus.modulo_matrices: given matrix b ' +\
            'is not a properly formated vector (has shape of {}).'.format(
                b.shape
            )
        )
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            'Minplus.modulo_matrices: given matrix b ' +\
            'does not have an Mx1 shape against MxN matrix A (A: {}, b: {}).'.format(
                A.shape, b.shape
            )
        )
    if np.any(A < 0) or np.any(b < 0):
        raise ValueError(
            'Minplus.modulo_matrices: matrices contain negative values.'
        )
    result = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i, j] = modulo(A[i, j], b[i])
    return result


def power(a : Number, k : int) -> Number:
    return mult(*[a for _ in range(k)])


def power_matrix(A : np.ndarray, k : int) -> np.ndarray:
    if np.any(np.diagonal(A) != 0):
        raise ValueError(
            'Minplus.power_matrix: matrix contains non-zero values on diagonal.'
        )
    if k == 0:
        result = np.eye(A.shape[0], A.shape[1])
        result[result == 0] = math.inf
        result[result == 1] = 0
    else:
        result = A.copy()
        for _ in range(k):
            result = mult_matrices(A, result)
    return result
