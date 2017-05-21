# -*- coding: utf-8 -*-

r"""
This module implements functions used to learn graphical representations for
data given sample observations from these variables
"""

import utils
import numpy as np
from scipy import spatial
from pyunlocbox import functions, solvers


def log_degree_barrier(X, dist_type='sqeuclidean', alpha=1, beta=1, step=0.5,
                       w0=None, maxit=1000, rtol=1e-5, retall=False,
                       verbosity='NONE'):
    r"""
    Learn graph by imposing a log barrier on the degrees

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} - \alpha 1^{T} \log{W1} + \beta \| W \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the log barrier
    beta : float, optional
        Regularization parameter controlling the density of the graph
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must have the same dimensions as z.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution on the optimization. Only returned if
        verbosity != 'NONE'.

    Notes
    -----
    This is the solver proposed in [Kalofolias, 2016] :cite:`kalofolias2016`.


    Examples
    --------
    >>> import learn_graph as lg
    >>> import networkx as nx
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import spatial
    >>> G_gt = nx.waxman_graph(100)
    >>> pos = nx.random_layout(G_gt)
    >>> coords = np.array(list(pos.values()))
    >>> def s1(x, y):
            return np.sin((2 - x - y)**2)
    >>> def s2(x, y):
            return np.cos((x + y)**2)
    >>> def s3(x, y):
            return (x - 0.5)**2 + (y - 0.5)**3 + x - y
    >>> def s4(x, y):
            return np.sin(3 * ( (x - 0.5)**2 + (y - 0.5)**2 ) )
    >>> X = np.array((s1(coords[:,0], coords[:,1]),
                      s2(coords[:,0], coords[:,1]),
                      s3(coords[:,0], coords[:,1]),
                      s4(coords[:,0], coords[:,1]))).T
    >>> z = 25 * spatial.distance.pdist(X, 'sqeuclidean')
    >>> W = lg.log_degree_barrier(z)
    >>> W[W < np.percentile(W, 96)] = 0
    >>> G_learned = nx.from_numpy_matrix(W)
    >>> plt.figure(figsize=(12, 6))
    >>> plt.subplot(1,2,1)
    >>> nx.draw(G_gt, pos=pos)
    >>> plt.title('Ground Truth')
    >>> plt.subplot(1,2,2)
    >>> nx.draw(G_learned, pos=pos)
    >>> plt.title('Learned')
    """

    # Parse X
    N = X.shape[0]
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Get primal-dual linear map
    K, Kt = utils.weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")
    stepsize = step / (1 + 2 * beta + norm_K)

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(
        np.finfo(np.float64).eps, K(w))))
    f2._prox = lambda d, gamma: np.maximum(
        0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gamma))))

    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w**2)
    f3._grad = lambda w: 2 * beta * w

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W
