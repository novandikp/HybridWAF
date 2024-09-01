import numpy as np
from numpy import ndarray
import random
from cloudpickle import dump
from data_evaluation import evaluation
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import os
import time
import pandas as pd


class NewPSOSVM:
    def __init__(self, W: float, c1: float, c2: float, n_particles: int, n_iterations: int):
        self.W = W
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.maxIter = 0
        self.keyFitnessFunction = "accuracy"
        self.best_model = None
        self.num_passes = 3
        self.tol = 0.001
        self.history = []

    def setData(self, x: ndarray, y: ndarray, val_size: float = 0.2, random_state: int = 42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=val_size, random_state=random_state
        )

    def setFitnessFunction(self, keyFitnessFunction: str):
        self.keyFitnessFunction = keyFitnessFunction

    def fitness_function(self, position: tuple) -> tuple:
        (gamma, c) = position
        if gamma <= 0:
            gamma = 0.0001
        if c <= 0:
            c = 0.0001

        svc = SVM(C=c, gamma=gamma)

        svc.fit(self.x_train, self.y_train)
        y_train_pred = svc.predict(self.x_train)
        y_test_pred = svc.predict(self.x_test)
        accuracyTrain, tprTrain, fprTrain, fdrTrain = evaluation(self.y_train, y_train_pred)
        accuracyTest, tprTest, fprTest, fdrTest = evaluation(self.y_test, y_test_pred)
        if self.keyFitnessFunction == "accuracy":
            score_train = 1 - accuracyTrain
            score_test = 1 - accuracyTest
        elif self.keyFitnessFunction == "tpr":
            score_train = 1 - tprTrain
            score_test = 1 - tprTest
        elif self.keyFitnessFunction == "fpr":
            score_train = fprTrain
            score_test = fprTest
        elif self.keyFitnessFunction == "fdr":
            score_train = fdrTrain
            score_test = fdrTest
        else:
            score_train = accuracyTrain
            score_test = accuracyTest
        return (
            score_train,
            score_test,
            svc,
        )

    def train(self, save_best: bool = True):
        start = time.time()
        particle_position_vector = np.array(
            [
                np.array([random.random() * 10, random.random() * 10])
                for _ in range(self.n_particles)
            ]
        )
        pbest_position = particle_position_vector
        pbest_fitness_value = np.array([float("inf") for _ in range(self.n_particles)])
        gbest_fitness_value = np.array([float("inf"), float("inf")])
        gbest_fitness_value_prev = np.array([float("inf"), float("inf")])
        gbest_position = np.array([float("inf"), float("inf")])
        velocity_vector = [np.array([0, 0]) for _ in range(self.n_particles)]
        iteration = 0
        passes = 0
        while iteration < self.n_iterations:
            for i in range(self.n_particles):
                print(f"particle- {i} iteration: {iteration}")
                train_res, test_res, model = self.fitness_function(
                    particle_position_vector[i]
                )
                fitness_cadidate = (train_res, test_res)
                print(f"error of particle- {i} is (training, test) "
                      f"{fitness_cadidate} At (gamma, c): "
                      f"{particle_position_vector[i]} Duration :{str(time.time() - start)} seconds")
                history_train = {
                    "iteration": iteration,
                    "particle": i,
                    "val_error": fitness_cadidate[1],
                    "train_error": fitness_cadidate[0],
                    "gamma": particle_position_vector[i][0],
                    "c": particle_position_vector[i][1],
                    "time": str(time.time() - start),
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
            for i in range(self.n_particles):
                new_velocity = (
                        (self.W * velocity_vector[i])
                        + (self.c1 * random.random())
                        * (pbest_position[i] - particle_position_vector[i])
                        + (self.c2 * random.random())
                        * (gbest_position - particle_position_vector[i])
                )
                new_position = new_velocity + particle_position_vector[i]
                particle_position_vector[i] = new_position
            if iteration > 0:
                improvement = gbest_fitness_value_prev[1] - gbest_fitness_value[1]
                if improvement <= self.tol and gbest_fitness_value_prev[1] > gbest_fitness_value[1]:
                    passes += 1
                    if passes >= self.num_passes:
                        break
                else:
                    passes = 0
            gbest_fitness_value_prev = gbest_fitness_value
            iteration += 1
            self.maxIter = iteration

    def predict(self, x: ndarray):
        return self.best_model.predict(x)

    def get_history(self):
        return self.history

    def getMaxIter(self):
        return self.maxIter

    def save_history(self, filename: str):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)

    def save_best(self, filename: str):
        dump(self.best_model, open(filename, "wb"))

    def __save_model(self, fitness_candicate: tuple, model: SVM, saved: bool):
        self.best_model = model
        if saved:
            # make directory if not exist
            if not os.path.exists("saved_model"):
                os.makedirs("saved_model")
            filename = f"saved_model/{fitness_candicate[1]}_{fitness_candicate[0]}.pkl"
            dump(self.best_model, open(filename, "wb"))
