from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class NonSquareError(Exception):
    pass

class NonSymmetricError(Exception):
    pass

class Matrix:

    """
    Internal data structure used to represent matrices. Can represent sparse, dense and diagonal matrices.

    Input
    -----
    A                   numpy 2D array
                        or scipy sparse array
                        or (N,) vector          rerpesents dia(vector)
                        or scalar               represents scalar * identity matrix of any size.
    assert_square       bool
    assert_symmetric    bool

    Methods:
    -------
    get_shape()                             returns matrix shape.
    is_diagonal()                           True if A is a diagonal matrix
    is_square()                             True if A is a square matrix
    diag()                                  returns diagonal of A. (if A is scalar; returns scalar)
    select(mask)                            returns A[mask, :][:, mask]
    stack(matrix: Matrix)                   returns [[self, 0], [0, matrix]]
    """

    def __init__(self,  A, assert_square=False, assert_symmetric=False):
        self.A = np.array(A) if not scipy.sparse.issparse(A) else A
        if self.A.ndim > 2:
            raise ValueError("matrix must be 2D")
        self.shape = None if self.A.ndim == 0 else self.A.shape
        if scipy.sparse.issparse(self.A):
            if self.shape[0] == self.shape[1]:
                if (self.A - scipy.sparse.diags(self.A.diagonal(), 0)).nnz == 0:
                    self.A = self.A.diagonal()
        if self.A.ndim == 1:
            if np.allclose(self.A, self.A[0]):
                self.A = self.A[0]
        self.is_scalar = self.A.ndim == 0
        if assert_square:
            if not self.is_diagonal():
                if not self.is_square():
                    raise NonSquareError("Matrix not square")
        if assert_symmetric:
            if not self.is_diagonal():
                if not self._is_symmetric(self.A):
                    raise NonSymmetricError("Matrix not symmetric")
        self.factorization = None

    def get_shape(self):
        if self.shape is None:
            return None
        if self.is_diagonal():
            return (self.shape[0], self.shape[0])
        return self.shape

    def matrix(self, N=None):
        if not self.is_diagonal():
            return self.A
        else:
            return scipy.sparse.diags(self.diag(force_as_vector=True, vector_length=N), 0)

    def is_diagonal(self):
        return self.A.ndim <= 1

    def is_zero(self):
        if self.is_diagonal():
            return np.all(self.diag() == 0)
        return False

    def is_square(self):
        shape = self.get_shape()
        return shape[0] == shape[1] if shape is not None else True

    def diag(self, force_as_vector=False, vector_length=None):
        out = self.A
        if not self.is_diagonal():
            out = self.A.diagonal()
        if not self.is_square():
            raise NonSquareError("Cannot extract diagonal; Matrix not square")
        return np.broadcast_to(out, (self.get_shape()[0] if vector_length is None else vector_length)) if force_as_vector else out

    def select(self, mask) -> Matrix:
        # returns A[mask, :][:, mask]
        if self.is_scalar:
            return Matrix(self.A)
        if self.is_diagonal():
            return Matrix(self.A[mask])
        return Matrix(self.A[mask, :][:, mask])

    def stack(self, matrix: Matrix) -> Matrix:
        # returns [[self, 0], [0, matrix]]
        if self.shape is None or matrix.shape is None:
            raise ValueError("cannot stack matrices of indeterminate size")
        N1, N2 = self.shape[0], matrix.shape[0]
        x1, x2 = self.A, matrix.A
        if self.is_scalar and matrix.is_scalar:
            if self.A == matrix.A:
                return Matrix(x1)
        if self.is_scalar:
            x1 = np.ones(N1, x1.dtype) * x1
        if matrix.is_scalar:
            x2 = np.ones(N2, x2.dtype) * x2
        if self.is_diagonal() and matrix.is_diagonal():
            return Matrix(np.append(x1, x2))
        if self.is_diagonal():
            x1 = scipy.sparse.diags(x1, 0)
        if matrix.is_diagonal():
            x2 = scipy.sparse.diags(x2, 0)
        return Matrix(scipy.sparse.block_diag((x1, x2)))

    def _is_symmetric(self, A):
        if scipy.sparse.isspmatrix(A):
            return (A - A.T).nnz == 0
        else:
            return np.all(A == A.T)

    def __str__(self):
        if self.is_scalar:
            return f"Matrix object representing {self.A} times identity matrix"
        if self.is_diagonal():
            return f"diagonal Matrix object of shape {self.get_shape()}"
        return f"Matrix object of shape {self.get_shape()}"
