import numpy as np
from pickle import dump
# from util.Notification import send_notification
from datetime import datetime


class SVM:
    def __init__(self, config, gamma=1, C=1, tol=1e-3, max_passes=50):
        self.C = float(C)
        self.gamma = float(gamma)
        self.kernel = self.rbf
        self.tol = tol
        # self.config = config
        self.max_passes = max_passes
        self.epsilon = 1e-5

    def rbf(self, Xi, Xj):
        return np.exp(-self.gamma * np.linalg.norm(Xi - Xj) ** 2)

    def fit(self, X, y):
        self.m_samples, self.n_features = X.shape
        self.X = X
        self.y = y.astype(np.float64)
        # assert self.X.shape[0] == self.y.shape[0]
        #
        # for i in range(self.m_samples):
        #     assert self.y[i] == 0 or self.y[i] == 1
        # changes classes to -1 and 1
        self.y = 2 * self.y - 1

        self.K = np.zeros((self.m_samples, self.m_samples))

        start_time = datetime.now()
        for i in range(self.m_samples):
            for j in range(self.m_samples):
                self.K[i, j] = self.kernel(self.X[i], self.X[j])
        print(f"Kernel matrix has been calculated in {datetime.now() - start_time}")
        # send_notification(self.config.NOTIFICATION,
        #                   f"Kernel matrix has been calculated in {datetime.now() - start_time}")

        def f(index):  # f(x) = w^T * x + b
            return np.sum(self.alphas * self.y * self.K[index, :]) + self.b

        self.alphas = np.zeros(shape=self.m_samples)
        self.b = 0
        passes = 0
        while (passes < self.max_passes):
            num_changed_alphas = 0
            half = int(self.m_samples / 2)
            for i in range(self.m_samples):
                E_i = f(i) - self.y[i]
                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (
                        self.y[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = np.random.choice(list(range(0, i)) + list(range(i + 1, self.m_samples)))
                    assert i != j
                    E_j = f(j) - self.y[j]
                    old_alpha_i, old_alpha_j = self.alphas[i], self.alphas[j]
                    L = max(0, self.alphas[j] - self.alphas[i]) if self.y[i] != self.y[j] else max(0, self.alphas[i] +
                                                                                                   self.alphas[
                                                                                                       j] - self.C)
                    H = min(self.C, self.C + self.alphas[j] - self.alphas[i]) if self.y[i] != self.y[j] else min(self.C,
                                                                                                                 self.alphas[
                                                                                                                     i] +
                                                                                                                 self.alphas[
                                                                                                                     j])
                    if L == H:
                        continue
                    ni = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if ni >= 0:
                        continue

                    alpha_j = self.alphas[j] - self.y[j] * (E_i - E_j) / ni
                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L

                    if abs(old_alpha_j - alpha_j) < self.tol:
                        continue

                    alpha_i = self.alphas[i] + self.y[i] * self.y[j] * (old_alpha_j - alpha_j)

                    b_1 = self.b - E_i - self.y[i] * (alpha_i - old_alpha_i) * self.K[i, i] - self.y[j] * (
                            alpha_j - old_alpha_j) * self.K[i, j]
                    b_2 = self.b - E_j - self.y[i] * (alpha_i - old_alpha_i) * self.K[i, j] - self.y[j] * (
                            alpha_j - old_alpha_j) * self.K[j, j]

                    if 0 < alpha_i < self.C:
                        b = b_1
                    elif 0 < alpha_j < self.C:
                        b = b_2
                    else:
                        b = (b_1 + b_2) / 2

                    num_changed_alphas += 1

                    self.alphas[i] = alpha_i
                    self.alphas[j] = alpha_j
                    self.b = b

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.support_ = np.arange(self.m_samples)[self.alphas > self.epsilon]
        self.support_vectors_ = self.X[self.support_]
        self.dual_coef_ = [self.alphas[self.support_]]

        b = sum(self.y[i] - sum(self.alphas[j] * self.y[j] * self.K[i, j]
                                for j in self.support_)
                for i in self.support_)
        if len(self.support_) > 0:
            b /= len(self.support_)
        else:
            b = 0

        self.b = b
        self.intercept_ = b

    def predict_one(self, x):
        w_phi_x = sum(self.alphas[i] * self.y[i] * self.kernel(self.X[i], x)
                      for i in self.support_)
        return w_phi_x + self.b

    def predictor(self, x):
        xs = [x] if len(x.shape) == 1 else x
        return np.array([self.predict_one(x) for x in xs])

    def predict(self, X):
        return (self.predictor(X) > 0).astype(np.float64)

    def save_model(self, path):
        dump(self, open(path, "wb"))
