# from datasets import getFormattedECMLDatasets, getFormattedCSICDatasets
# from feature_extraction import transform_data
from data_training.NewPSOSVM import NewPSOSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler
from data_evaluation import evaluation
import os
import pandas as pd

print("Feature extraction...")
# dataECML = getFormattedECMLDatasets()
# featureECML = transform_data(dataECML)
# labelEcml = dataECML['type']
# minMaxECML = MinMaxScaler()
# minMaxECML.fit(featureECML)
# featureECML = minMaxECML.transform(featureECML)
#
# # save to csv
# dataUpdatedECML = pd.concat([pd.DataFrame(featureECML), labelEcml], axis=1)
# dataUpdatedECML.to_csv("dataUpdatedECML.csv", index=False)
#
# dataCSIC = getFormattedCSICDatasets()
# labelCsic = dataCSIC['type']
# dataCSIC = transform_data(dataCSIC)
# minMaxCSIC = MinMaxScaler()
# minMaxCSIC.fit(dataCSIC)
# featureCSIC = minMaxCSIC.transform(dataCSIC)
#
# # save to csv
# dataUpdatedCSIC = pd.concat([pd.DataFrame(featureCSIC), labelCsic], axis=1)
# dataUpdatedCSIC.to_csv("dataUpdatedCSIC.csv", index=False)
#

featureECML = pd.read_csv("dataUpdatedECML.csv")
labelEcml = featureECML['type']
featureECML = featureECML.drop(columns=['type'])

featureCSIC = pd.read_csv("dataUpdatedCSIC.csv")
# filter is null
featureCSIC = featureCSIC[featureCSIC['type'].notnull()]
labelCsic = featureCSIC['type']
featureCSIC = featureCSIC.drop(columns=['type'])


def run_scenario(scenario, tryCount):

    X_TrainECML, X_TestECML, Y_TrainECML, Y_TestECML = train_test_split(featureECML, labelEcml,
                                                                        test_size=scenario.test_size)
    X_TrainCSIC, X_TestCSIC, Y_TrainCSIC, Y_TestCSIC = train_test_split(featureCSIC, labelCsic,
                                                                        test_size=scenario.test_size)

    fileNameECML = "ECML_ts#" + str(scenario.test_size) + "_w#" + str(scenario.W) + "_c1#" + str(
        scenario.c1) + "_c2#" + str(scenario.c2) + "_try#" + str(tryCount) + ".pkl"
    print(fileNameECML)
    fileNameHistoryECML = "ECML_ts#" + str(scenario.test_size) + "_w#" + str(scenario.W) + "_c1#" + str(
        scenario.c1) + "_c2#" + str(scenario.c2) + "_try#" + str(tryCount) + ".csv"
    psoECML = NewPSOSVM(scenario.W, scenario.c1, scenario.c2, scenario.n_particles, scenario.n_iterations)
    psoECML.setData(X_TrainECML, Y_TrainECML)
    psoECML.setFitnessFunction(scenario.fitness)
    psoECML.train()

    fileNameCSIC = "CSIC_ts#" + str(scenario.test_size) + "_w#" + str(scenario.W) + "_c1#" + str(
        scenario.c1) + "_c2#" + str(scenario.c2) + "_try#" + str(tryCount) + ".pkl"
    print(fileNameCSIC)
    fileNameHistoryCSIC = "CSIC_ts#" + str(scenario.test_size) + "_w#" + str(scenario.W) + "_c1#" + str(
        scenario.c1) + "_c2#" + str(scenario.c2) + "_try#" + str(tryCount) + ".csv"
    psoCSIC = NewPSOSVM(scenario.W, scenario.c1, scenario.c2, scenario.n_particles, scenario.n_iterations)
    psoCSIC.setData(X_TrainCSIC, Y_TrainCSIC)
    psoCSIC.setFitnessFunction(scenario.fitness)
    psoCSIC.train()

    y_predECML = psoECML.predict(X_TestECML)
    y_predCSIC = psoCSIC.predict(X_TestCSIC)

    if not os.path.exists("new_model"):
        os.makedirs("new_model")
    fileNameECML = os.path.join("new_model", fileNameECML)
    fileNameCSIC = os.path.join("new_model", fileNameCSIC)
    psoECML.save_best(fileNameECML)
    psoCSIC.save_best(fileNameCSIC)

    if not os.path.exists("new_history"):
        os.makedirs("new_history")
    fileNameHistoryECML = os.path.join("new_history", fileNameHistoryECML)
    fileNameHistoryCSIC = os.path.join("new_history", fileNameHistoryCSIC)

    psoECML.save_history(fileNameHistoryECML)
    psoCSIC.save_history(fileNameHistoryCSIC)

    accuracyECML, tprECML, fprECML, fdrECML = evaluation(Y_TestECML, y_predECML)
    accuracyCSIC, tprCSIC, fprCSIC, fdrCSIC = evaluation(Y_TestCSIC, y_predCSIC)

    crECML = cm(Y_TestECML, y_predECML)
    crCSIC = cm(Y_TestCSIC, y_predCSIC)

    tpECML = crECML[1][1]
    tnECML = crECML[0][0]
    fpECML = crECML[0][1]
    fnECML = crECML[1][0]

    tpCSIC = crCSIC[1][1]
    tnCSIC = crCSIC[0][0]
    fpCSIC = crCSIC[0][1]
    fnCSIC = crCSIC[1][0]

    maxIterECML = psoECML.getMaxIter()
    maxIterCSIC = psoCSIC.getMaxIter()

    result = {
        "test_size": scenario.test_size,
        "W": scenario.W,
        "c1": scenario.c1,
        "c2": scenario.c2,
        "fitness": scenario.fitness,
        "n_particles": scenario.n_particles,
        "n_iterations": scenario.n_iterations,
        "maxIterECML": maxIterECML,
        "maxIterCSIC": maxIterCSIC,
        "tpECML": tpECML,
        "tnECML": tnECML,
        "fpECML": fpECML,
        "fnECML": fnECML,
        "accuracyECML": accuracyECML,
        "tprECML": tprECML,
        "fprECML": fprECML,
        "fdrECML": fdrECML,
        "tpCSIC": tpCSIC,
        "tnCSIC": tnCSIC,
        "fpCSIC": fpCSIC,
        "fnCSIC": fnCSIC,
        "accuracyCSIC": accuracyCSIC,
        "tprCSIC": tprCSIC,
        "fprCSIC": fprCSIC,
        "fdrCSIC": fdrCSIC,
        "tryCount": tryCount
    }

    history_scenario.append(result)
    return result


def run_three_scenario(scenario):
    tryCount = 1
    resultScene = []
    while tryCount <= 3:
        result = run_scenario(scenario, tryCount)
        # calculate the average of accuracy, tpr, fpr, fdr
        resultScene.append(result)
        tryCount += 1

    avg_accuracyECML = sum([result["accuracyECML"] for result in resultScene]) / len(resultScene)
    avg_accuracyCSIC = sum([result["accuracyCSIC"] for result in resultScene]) / len(resultScene)
    avg_tprECML = sum([result["tprECML"] for result in resultScene]) / len(resultScene)
    avg_tprCSIC = sum([result["tprCSIC"] for result in resultScene]) / len(resultScene)
    avg_fprECML = sum([result["fprECML"] for result in resultScene]) / len(resultScene)
    avg_fprCSIC = sum([result["fprCSIC"] for result in resultScene]) / len(resultScene)
    avg_fdrECML = sum([result["fdrECML"] for result in resultScene]) / len(resultScene)
    avg_fdrCSIC = sum([result["fdrCSIC"] for result in resultScene]) / len(resultScene)

    avg_accuracy = (avg_accuracyECML + avg_accuracyCSIC) / 2
    avg_tpr = (avg_tprECML + avg_tprCSIC) / 2
    avg_fpr = (avg_fprECML + avg_fprCSIC) / 2
    avg_fdr = (avg_fdrECML + avg_fdrCSIC) / 2

    history_summary.append({
        "test_size": scenario.test_size,
        "W": scenario.W,
        "c1": scenario.c1,
        "c2": scenario.c2,
        "fitness": scenario.fitness,
        "n_particles": scenario.n_particles,
        "n_iterations": scenario.n_iterations,
        "avg_accuracy": avg_accuracy,
        "avg_tpr": avg_tpr,
        "avg_fpr": avg_fpr,
        "avg_fdr": avg_fdr
    })
    return avg_accuracy


scenarios = {
    "test_size": [0.15, 0.2, 0.25],
    "W": [0.5, 1, 1.5],
    "c1": [1, 2, 3],
    "c2": [1, 2, 3],
}


class Scenario:
    def __init__(self, test_size, W, c1, c2, n_particles=2, n_iterations=2, fitness="accuracy", score=0):
        self.test_size = test_size
        self.W = W
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.fitness = fitness
        self.score = score


current_scenario = Scenario(
    test_size=scenarios["test_size"][0],
    W=scenarios["W"][0],
    c1=scenarios["c1"][0],
    c2=scenarios["c2"][0],
    n_particles=15,
    n_iterations=20,
    fitness="accuracy",
    score=0
)
print("Current scenario: ", current_scenario)
best_scenario = current_scenario
history_scenario = []
history_summary = []

# for scenario in scenarios:
print("Running scenario... test_size")
for test_size in scenarios.get("test_size"):
    current_scenario.__setattr__("test_size", test_size)
    score = run_three_scenario(current_scenario)
    if score > best_scenario.score:
        best_scenario = current_scenario
        best_scenario.score = score
        # print best test_size with score
        print("Best test_size: ", test_size, " with score: ", score)

current_scenario.test_size = best_scenario.test_size
print("Running scenario... W")
# for w in scenarios["W"]:
for w in scenarios["W"]:
    # skip first iteration
    if w == scenarios["W"][0]:
        continue
    current_scenario.W = w
    score = run_three_scenario(current_scenario)
    print("Score: ", score)
    print("Previous best score: ", best_scenario.score)
    if score > best_scenario.score:
        best_scenario = current_scenario
        best_scenario.score = score
        # print best w with score
        print("Best W: ", w, " with score: ", score)


current_scenario.W = best_scenario.W

# for c1 in scenarios["c1"]:
print("Running scenario... c1")
for c1 in scenarios["c1"]:
    # skip first iteration
    if c1 == scenarios["c1"][0]:
        continue
    current_scenario.c1 = c1
    score = run_three_scenario(current_scenario)
    if score > best_scenario.score:
        best_scenario = current_scenario
        best_scenario.score = score
        print("Best C1: ", c1, " with score: ", score)

current_scenario.c1 = best_scenario.c1
# for c2 in scenarios["c2"]:
print("Running scenario... c2")
for c2 in scenarios["c2"]:
    # skip first iteration
    if c2 == scenarios["c2"][0]:
        continue
    current_scenario.c2 = c2
    score = run_three_scenario(current_scenario)
    if score > best_scenario.score:
        best_scenario = current_scenario
        best_scenario.score = score
        print("Best C2: ", c2, " with score: ", score)

current_scenario.c2 = best_scenario.c2
# sace history
history_scenario = pd.DataFrame(history_scenario)
history_summary = pd.DataFrame(history_summary)
if not os.path.exists("new_history"):
    os.makedirs("new_history")
history_scenario.to_csv(os.path.join("new_history", "history_scenario.csv"), index=False)
history_summary.to_csv(os.path.join("new_history", "history_summary.csv"), index=False)