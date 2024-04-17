import numpy as np
from numpy import ndarray
from cloudpickle import dump
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_random_state
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level="3")
from . import _svm


class SVM():
    """Support vector machine (SVM).

    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm"
    (http://math.unt.edu/~hsp0009/smo.pdf)
    Based on Karpathy's svm.js implementation:
    https://github.com/karpathy/svmjs

    Parameters
    ----------
    C : float, optional (default: 1)
        Penalty parameter C of the error term.

    kernel : string, optional (default: 'rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``

    degree : int, optional (default: 3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default: 1)
        Parameter for RBF kernel


    tol : float, optional (default: 1e-4)
        Numerical tolerance. Usually this should not be modified.


    maxiter : int, optional (default: 10000)
        Maximum number of iterations

    numpasses : int, optional (default: 10)
        How many passes over data with no change before we halt? Increase for
        more precision.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    support_vectors_ : array-like, shape = [n_support_vectors, n_features]
        Support vectors.

    coef_ : array, shape = [n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : float
        Constant in decision function.
    """

    def __init__(self,
                 C: float = 1.0, gamma: float = 1.0,
                 tol: float = 1e-4, maxiter: int = 30, numpasses: int = 5,
                 random_state=None, verbose=0):
        self.C = C
        self.kernel = "rbf"
        self.gamma = gamma
        self.tol = tol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: ndarray, y: ndarray):
        """Fit the model to data matrix X and target y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The class labels.

        Returns
        -------
        self : returns a trained SVM
        """
        #  convert X and y is float
        X = X.astype(float)
        y = y.astype(float)

        self.support_vectors_ = check_array(X)
        self.y = check_array(y, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        self.alpha_ = np.zeros(X.shape[0])
        self.intercept_ = _svm.smo(
            K, y, self.alpha_, self.C, random_state, self.tol,
            self.numpasses, self.maxiter, self.verbose)

        # only samples with nonzero coefficients are relevant for predictions
        support_vectors = np.nonzero(self.alpha_)
        self.alpha_ = self.alpha_[support_vectors]
        self.support_vectors_ = X[support_vectors]
        self.y = y[support_vectors]

        return self

    def decision_function(self, X: ndarray) -> ndarray:
        """Decision function of the SVM.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The values of decision function.
        """
        X = check_array(X)
        K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel,
                             gamma=self.gamma)
        return (self.intercept_ + np.sum(self.alpha_[np.newaxis, :] *
                                         self.y * K, axis=1))

    def predict(self, X: ndarray) -> ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The predicted classes.
        """
        return np.sign(self.decision_function(X))

    def save_model(self, path: str):
        dump(self, open(path, "wb"))
