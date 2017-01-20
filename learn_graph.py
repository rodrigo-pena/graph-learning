# -*- coding: utf-8 -*-

r"""
This module implements functions used to learn graphical representations for
data given sample observations from these variables
"""

import utils
import numpy as np
from scipy import spatial
from pyunlocbox import functions, solvers


def log_degree_barrier(z, alpha=1, beta=1, step=0.5, w0=None, maxit=1000,
                       rtol=1e-5, verbosity='NONE'):
    r"""
    Learn graph by imposing a log barrier on the degrees

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} - \alpha 1^{T} \log{W1} + \beta \| W \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.

    Parameters
    ----------
    z : array_like
        A vector of dimension N(N - 1)/2 or a symmetric matrix of dimension
        NxN encoding the pairwise distances between nodes.
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

    See :func:`scipy.spatial.distance` for examples on how to generate
    pairwise distances.

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

    # Parse z
    z, N = utils.parse_distances(z)

    # Get primal-dual linear map
    K, Kt = utils.weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Parse stepsize
    stepsize = step / (2 * beta + norm_K)
    stepsize = min(stepsize, 1 / (2 * beta + norm_K))
    stepsize = max(stepsize, 0)

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    assert w0.shape == z.shape, "w0 must have the same shape as z."

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

    if verbosity == 'NONE':
        return spatial.distance.squareform(problem['sol'])
    else:
        return spatial.distance.squareform(problem['sol']), problem
