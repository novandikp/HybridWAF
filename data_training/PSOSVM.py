import numpy as np
import random
from util.Notification import send_notification
from data_training.SVM import SVM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd


class PSOSVM:
    def __init__(self, config):
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.best_model = None
        self.history = []
        self.config = config

    def setData(self, x, y, val_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=val_size, random_state=random_state
        )

    def fitness_function(self, position):
        (gamma, c) = position
        # Check if gamma and c is not negative or zero
        if gamma <= 0:
            gamma = 0.0001
        if c <= 0:
            c = 0.0001

        # svc = SVC(kernel="rbf", gamma=gamma, C=c)
        svc = SVM(self.config, gamma=gamma, C=c)
        svc.fit(self.x_train, self.y_train)
        y_train_pred = svc.predict(self.x_train)
        y_test_pred = svc.predict(self.x_test)
        return (
            1 - accuracy_score(self.y_train, y_train_pred),
            1 - accuracy_score(self.y_test, y_test_pred),
            svc,
        )

    def train(self, save_best=True):
        start = time.time()
        particle_position_vector = np.array(
            [
                np.array([random.random() * 10, random.random() * 10])
                for _ in range(self.config.PSO.n_particles)
            ]
        )
        pbest_position = particle_position_vector
        pbest_fitness_value = np.array([float("inf") for _ in range(self.config.PSO.n_particles)])
        gbest_fitness_value = np.array([float("inf"), float("inf")])
        gbest_fitness_value_prev = np.array([float("inf"), float("inf")])
        gbest_position = np.array([float("inf"), float("inf")])
        velocity_vector = [np.array([0, 0]) for _ in range(self.config.PSO.n_particles)]
        iteration = 0

        while iteration < self.config.PSO.n_iterations:
            for i in range(self.config.PSO.n_particles):
                send_notification(self.config.NOTIFICATION,f"particle- {i} iteration: {iteration}")
                train_res, test_res, model = self.fitness_function(
                    particle_position_vector[i]
                )
                fitness_cadidate = (train_res, test_res)
                send_notification(self.config.NOTIFICATION, f"error of particle- {i} is (training, test) "
                                                            f"{fitness_cadidate} At (gamma, c): "
                                                            f"{particle_position_vector[i]} Duration :{str(time.time() - start)} seconds")
                history_train = {
                    "iteration": iteration,
                    "particle": i,
                    "val_error": fitness_cadidate[1],
                    "train_error": fitness_cadidate[0],
                    "gamma": particle_position_vector[i][0],
                    "c": particle_position_vector[i][1],
                }
                self.history.append(history_train)
                if pbest_fitness_value[i] > fitness_cadidate[1]:
                    pbest_fitness_value[i] = fitness_cadidate[1]
                    pbest_position[i] = particle_position_vector[i]

                if gbest_fitness_value[1] > fitness_cadidate[1]:
                    gbest_fitness_value = fitness_cadidate
                    gbest_position = particle_position_vector[i]
                    self.__save_model(fitness_cadidate, model, save_best)
                elif (
                        gbest_fitness_value[1] == fitness_cadidate[1]
                        and gbest_fitness_value[0] > fitness_cadidate[0]
                ):
                    gbest_fitness_value = fitness_cadidate
                    gbest_position = particle_position_vector[i]
                    self.__save_model(fitness_cadidate, model, save_best)
            for i in range(self.config.PSO.n_particles):
                new_velocity = (
                        (self.config.PSO.W * velocity_vector[i])
                        + (self.config.PSO.c1 * random.random())
                        * (pbest_position[i] - particle_position_vector[i])
                        + (self.config.PSO.c2 * random.random())
                        * (gbest_position - particle_position_vector[i])
                )
                new_position = new_velocity + particle_position_vector[i]
                particle_position_vector[i] = new_position
            if iteration > 0 and self.config.PSO.tol > 0:
                diff = np.subtract(gbest_fitness_value, gbest_fitness_value_prev)
                diff = np.abs(diff)
                if np.all(diff < self.config.PSO.tol):
                    send_notification(self.config.NOTIFICATION,
                                      f"Converged at iteration {iteration} with convergence value {diff}")
                    break

            gbest_fitness_value_prev = gbest_fitness_value
            iteration = iteration + 1

    def predict(self, x):
        return self.best_model.predict(x)

    def get_history(self):
        return self.history

    def save_history(self, filename):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)

    def save_best(self, filename):
        self.best_model.save_model(filename)

    def __save_model(self, fitness_candicate, model, saved):
        self.best_model = model
        if saved:
            # make directory if not exist
            if not os.path.exists("saved_model"):
                os.makedirs("saved_model")
            filename = f"saved_model/{fitness_candicate[1]}_{fitness_candicate[0]}.pkl"
            self.best_model.save_model(filename)


