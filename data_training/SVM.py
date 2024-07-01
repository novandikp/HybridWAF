import numpy as np
from numpy import ndarray
from cloudpickle import dump
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_random_state
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level="3")
from . import _svm


class SVM:

    def __init__(self,
                 C: float = 1.0, gamma: float = 1.0,
                 tol: float = 1e-4, maxiter: int = 50, numpasses: int = 5,
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

        if self.verbose >= 2:
            print("Intercept: ", self.intercept_)
        support_vectors = np.nonzero(self.alpha_)
        self.alpha_ = self.alpha_[support_vectors]
        self.support_vectors_ = X[support_vectors]
        self.y = y[support_vectors]
        return self

    def decision_function(self, X: ndarray) -> ndarray:
        X = check_array(X)
        K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel,
                             gamma=self.gamma)
        return (self.intercept_ + np.sum(self.alpha_[np.newaxis, :] *
                                         self.y * K, axis=1))

    def predict(self, X: ndarray) -> ndarray:
        return np.sign(self.decision_function(X))

    def save_model(self, path: str):
        dump(self, open(path, "wb"))
