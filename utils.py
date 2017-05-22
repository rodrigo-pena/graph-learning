# -*- coding: utf-8 -*-

r"""
This module defines helper functions

Several of these functions were based on the pygsp toolbox:
http://pygsp.readthedocs.io/en/latest/
"""

import numpy as np
from scipy import sparse

# -----------------------------------------------------------------------------
""" Functions used in :func:`learn_graph.log_degree_barrier method` """


def weight2degmap(N, array=False):
    r"""
    Generate linear operator K such that W @ 1 = K @ vec(W).

    Parameters
    ----------
    N : int
        Number of nodes on the graph

    Returns
    -------
    K : function
        Operator such that K(w) is the vector of node degrees
    Kt : function
        Adjoint operator mapping from degree space to edge weight space
    array : boolean, optional
        Indicates if the maps are returned as array (True) or callable (False).

    Examples
    --------
    >>> import learn_graph
    >>> K, Kt = learn_graph.weight2degmap(10)

    Notes
    -----
    Used in :func:`learn_graph.log_degree_barrier method`.

    """
    import numpy as np

    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne, ))
    row_idx2 = np.zeros((Ne, ))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count: (count + (N - i))] = i - 1
        row_idx2[count: (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)


def plot_objectives(objective, labels=None, fig=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm

    objective = np.asarray(objective)

    n = objective.shape[0]
    try:
        m = objective.shape[1]
        obj = objective
    except IndexError:
        m = 1
        obj = np.reshape(objective, (n, 1))

    if labels is None:
        labels = []
        for i in range(m):
            labels.append('obj. ' + str(i + 1))
    assert m == len(labels), "Must have same number of labels as obj. fun."

    if fig is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = fig.axes[-1]

    fig.set_figheight(6)
    fig.set_figwidth(12)

    stride = 1. / (m - 1) if m > 1 else 1.
    cmap = matplotlib.cm.get_cmap('viridis')

    for i in range(m):
        color = cmap(i * stride)
        ax.plot(obj[:, i], color=color, linewidth=2, label=labels[i])

    ax.legend(loc='best')
    ax.set_xlim([-1, n + 1])
    ax.set_ylim([np.min(obj) - 1, np.max(obj) + 1])

    return fig, ax


# -----------------------------------------------------------------------------


def deal_with_sparse(M, sparse_flag):
    r""" Make M coherent with sparse_flag """
    if sparse_flag and not sparse.issparse(M):
        return sparse.csr_matrix(M)
    elif not sparse_flag and sparse.issparse(M):
        return M.toarray()
    else:
        return M


def estimate_lmax(L, sparse_flag=True, dummy=False):
    r""" Estimate the maximal eigenvalue of matrix L. """

    if dummy:
        return 2.0

    if L.shape[0] == 0:
        return

    L = deal_with_sparse(L, sparse_flag)

    try:
        if sparse_flag:
            lmax = sparse.linalg.eigs(L, k=1, tol=5e-3, ncv=10)[0][0]
        else:
            lmax = np.max(np.linalg.eigvals(L))
    except:
        lmax = 2. * np.max(L.diagonal())

    lmax = np.real(lmax)
    return lmax.sum()


def create_laplacian(W, lap_type='combinatorial', deg_type='out',
                     sparse_flag=True):
    r"""
    Create a graph laplacian matrix.

    Parameters
    ----------
    W : array
        Adjacency matrix
    lap_type : string
        Laplacian type to use ['combinatorial','random-walk', 'normalized'].
        Default:'combinatorial'.
    deg_type : string
        Degree type to use in case the graph is directed ['in', 'out',
        'average']. Default: 'out'.
    sparse_flag : bool
        Use sparse matrices (True) or not (False).

    """
    n, _ = W.shape

    if n == 0:
        return sparse.lil_matrix((0, 0)) if sparse_flag else np.array([])
    if n == 1:
        return sparse.lil_matrix(0) if sparse_flag else np.array([0])

    W = deal_with_sparse(W, sparse_flag)

    if lap_type not in ['combinatorial', 'normalized', 'random-walk']:
        raise AttributeError('Unknown laplacian type!')

    if deg_type not in ['in', 'out', 'average']:
        raise AttributeError('Unknown degree type!')

    d = deg_vec(W, deg_type, sparse_flag)
    D = sparse.diags(d, 0).tocsr() if sparse_flag else np.diag(d)
    L = (D - W).tocsr() if sparse_flag else D - W

    if lap_type == 'random-walk':
        if sparse_flag:
            D = sparse.diags(div0(np.ones(n), d), 0).tocsr()
        else:
            D = np.linalg.pinv(D)
        L = np.dot(D, L)
    elif lap_type == 'normalized':
        if sparse_flag:
            D = sparse.diags(div0(np.ones(n), d)**(.5), 0).tocsr()
        else:
            D = np.power(np.linalg.pinv(D), 0.5)
        L = np.dot(D, np.dot(L, D))

    return L


def is_directed(W):
    r"""
    Define if the graph has directed edges.

    Parameters
    ---------
    W: array
        Adjacency matrix

    Notes
    -----
    Can also be used to check if a matrix is symmetrical

    """
    if np.diff(np.shape(W))[0]:
        raise ValueError("Expected square matrix.")

    return np.abs(W - W.T).sum() != 0


def deg_vec(W, deg_type='out', sparse_flag=True):
    r"""
    Create the degree vector

    Parameters
    ----------
    W : array
        Adjacency matrix
    deg_type : string
        Degree type to use in case the graph is directed.
    sparse_flag : bool
        Use sparse matrices (True) or not (False).

    """
    if is_directed(W):
        if deg_type == 'in':
            d = sparse.tril(W, k=0, format='csr').sum(
                1) if sparse_flag else np.sum(np.tril(W), axis=1)
        elif deg_type == 'out':
            d = sparse.triu(W, k=0, format='csr').sum(
                1) if sparse_flag else np.sum(np.triu(W), axis=1)
        elif deg_type == 'average':
            d = 0.5 * (np.sum(W, axis=0) + np.sum(W, axis=1))
    else:
        d = np.sum(W, axis=1)

    return np.ravel(d)


def div0(a, b):
    r"""
    Elementwise division, returning zero when dividing by zero.

    Parameters
    ----------
    a : array_like
    b : array_like

    """
    x = np.ravel(np.array(a))
    y = np.ravel(np.array(b))
    assert len(x) == len(y), \
        "Input arrays must have the same number of elements"
    z = []
    for i in range(len(y)):
        if y[i] != 0:
            z = np.concatenate((z, [x[i] / y[i]]))
        else:
            z = np.concatenate((z, [0.]))
    return np.reshape(z, np.shape(a))


def n_edges(n, directed=False, self_loops=False):
    r""" Compute the max number of edges on a graph with n nodes """
    if directed:
        if self_loops:
            ne = n ** 2
        else:
            ne = n * (n - 1)
    else:
        if self_loops:
            ne = n * (n + 1) / 2
        else:
            ne = n * (n - 1) / 2

    return int(ne)


def cheby_op(L, c, X, lmax=None, lmin=None, sparse_flag=True):
    r"""
    Chebyshev polynomial of a matrix, applied to a matrix/vector.

    Parameters
    ----------
    L : narray
        Matrix on which to evaluate the polynomial.
    c : ndarray
        Coefficients of the Chebyshev polynomial.
    X : ndarray
        Vector or matrix to be multiplied by the matrix polynomial.
    lmax : float
        Estimative of the largest eigenvalue of L. It will be computed if not
        provided.
    lmin : float
        Estimative of the smallest eigenvalue of L. It will be computed if not
        provided.
    sparse_flag : bool
        Use sparse matrices (True) or not (False).

    Returns
    -------
    R : ndarray
        Result of the matrix multiplication

    """
    m = c.shape[0]  # Degree of polynomial + 1

    if m < 2:
        raise TypeError("The degree of the polynomial should be > 1")

    row, col = np.shape(L)

    if lmax is None:  # Estimate max eigenvalue
        lmax = estimate_lmax(L, sparse_flag)

    if lmin is None:  # Estimate min eigenvalue
        lmin = lmax - estimate_lmax(lmax * sparse.eye(row, col) -
                                    L, sparse_flag=sparse_flag)

    try:  # Allocate memory for R
        n, nv = np.shape(X)
        R = np.zeros((n, nv))
    except ValueError:
        n = np.shape(X)
        R = np.zeros((n))

    # Polynomial variable correction
    a = float(lmax - lmin) / 2.  # Scale coefficient
    b = float(lmax + lmin) / 2.  # Offset coefficient
    if sparse_flag:
        C = (L - b * sparse.eye(row, col)) / a
    else:
        C = (L - b * np.eye(row, col)) / a

    T_2 = X
    T_1 = np.dot(C, X)
    R = 0.5 * c[0] * T_2 + c[1] * T_1  # First iteration

    for k in range(2, m):
        T = 2 * np.dot(C, T_1) - T_2
        R += c[k] * T
        T_2 = T_1
        T_1 = T

    return R


def symmetrize(W, symm_type='average', sparse_flag=True):
    r"""
    Symmetrize a sparse square matrix

    Parameters
    ----------
    W : array_like
        Square matrix to be symmetrized
    symm_type : string
        'average' : symmetrize by averaging with the transpose.
        'full' : symmetrize by filling in the holes in the transpose.
    sparse_flag : bool
        Use sparse matrices (True) or not (False).

    """
    assert W.shape[0] == W.shape[1], "W must be a square matrix"

    W = deal_with_sparse(W, sparse_flag)

    if symm_type == 'average':
        return (W + W.T) / 2.
    elif symm_type == 'full':
        A = (W > 0)
        if sparse_flag:
            mask = ((A + A.T) - A).astype('float')
        else:
            mask = np.logical_xor(np.logical_or(A, A.T), A).astype('float')
        return W + mask.multiply(W.T) if sparse_flag else W + (mask * W.T)
    else:
        return W
