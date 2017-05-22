# -*- coding: utf-8 -*-

r"""
This module implements functions used to learn graphical representations for
data given sample observations from these variables
"""

import utils
import time
import numpy as np
from scipy import spatial
from sklearn.covariance import graph_lasso
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
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
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
        Information about the solution of the optimization. Only returned if
        retall == True.

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

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K, Kt = utils.weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

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
    lipg = 2 * beta

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

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


def l2_degree_reg(X, dist_type='sqeuclidean', alpha=1, s=None, step=0.5,
                  w0=None, maxit=1000, rtol=1e-5, retall=False,
                  verbosity='NONE'):
    r"""
    Learn graph by regularizing the l2-norm of the degrees.

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} + \alpha \|W1}\|^2 + \alpha \| W \|_{F}^{2}`, subject
    to :math:`\|W\|_{1,1} = s`, where :math:`Z` is a pairwise distance matrix,
    and :math:`\mathcal{W}_m`is the set of valid symmetric weighted adjacency
    matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the l2-norm.
    s : float, optional
        The "sparsity level" of the weight matrix, as measured by its l1-norm.
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
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
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This is the problem proposed in [Dong et al., 2015].


    Examples
    --------

    """

    # Parse X
    N = X.shape[0]
    E = int(N * (N - 1.) / 2.)
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse s
    s = N if s is None else s

    # Parse step
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    one_vec = np.ones(E)

    def K(w):
        return np.array([2 * np.dot(one_vec, w)])

    def Kt(n):
        return 2 * n * one_vec

    norm_K = 2 * np.sqrt(E)

    # Get weight-to-degree map
    S, St = utils.weight2degmap(N)

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: 0.
    f2._prox = lambda d, gamma: s

    f3 = functions.func()
    f3._eval = lambda w: alpha * (2 * np.sum(w**2) + np.sum(S(w)**2))
    f3._grad = lambda w: alpha * (4 * w + St(S(w)))
    lipg = 2 * alpha * (N + 1)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

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


def glasso(X, alpha=1, w0=None, maxit=1000, rtol=1e-5, retall=False,
           verbosity='NONE'):
    r"""
    Learn graph by imposing promoting sparsity in the inverse covariance.

    This is done by solving
    :math:`\tilde{W} = \underset{W \succeq 0}{\text{arg}\min} \,
    -\log \det W - \text{tr}(SW) + \alpha\|W \|_{1,1},
    where :math:`S` is the empirical (sample) covariance matrix.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    alpha : float, optional
        Regularization parameter acting on the l1-norm
    w0 : array_like, optional
        Initialization of the inverse covariance. Must be an N-by-N symmetric
        positive semi-definite matrix.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. If the dual gap goes below this value, iterations
        are stopped. See :func:`sklearn.covariance.graph_lasso`.
    retall : boolean
        Return solution and problem details.
    verbosity : {'NONE', 'ALL'}, optional
        Level of verbosity of the solver.
        See :func:`sklearn.covariance.graph_lasso`/

    Returns
    -------
    W : array_like
        Learned inverse covariance matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This function uses the solver :func:`sklearn.covariance.graph_lasso`.

    Examples
    --------

    """

    # Parse X
    S = np.cov(X)

    # Parse initial point
    w0 = np.ones(S.shape) if w0 is None else w0
    if (w0.shape != S.shape):
        raise ValueError("w0 must be of dimension N-by-N.")

    # Solve problem
    tstart = time.time()
    res = graph_lasso(emp_cov=S,
                      alpha=alpha,
                      cov_init=w0,
                      mode='cd',
                      tol=rtol,
                      max_iter=maxit,
                      verbose=(verbosity == 'ALL'),
                      return_costs=True,
                      return_n_iter=True)

    problem = {'sol':       res[1],
               'dual_sol':  res[0],
               'solver':    'sklearn.covariance.graph_lasso',
               'crit':      'dual_gap',
               'niter':     res[3],
               'time':      time.time() - tstart,
               'objective': np.array(res[2])[:, 0]}

    W = problem['sol']

    if retall:
        return W, problem
    else:
        return W
