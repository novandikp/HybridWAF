import datasets
import feature_extraction as fe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_training.PSOSVM import PSOSVM
from data_training.SVM import SVM
from util.Notification import send_notification, send_model
from data_evaluation import send_classification_report, evaluation
from util.LabelEncoder import LabelEncoder
from cloudpickle import dump
import os
import json
from datetime import datetime


def train(test_size: float = 0.25, dataset_type: str = "ecml", algorithm: str = "psosvm", fitness: str = "accuracy",
          config=None):
    send_notification(config.NOTIFICATION,
                      f"Training with test size: {test_size}, dataset: {dataset_type}, algorithm: {algorithm}")
    send_notification(config.NOTIFICATION, "Loading dataset...")
    if dataset_type == "ecml":
        dataset = datasets.getFormattedECMLDatasets()
    elif dataset_type == "csic":
        dataset = datasets.getFormattedCSICDatasets()
    else:
        raise ValueError("Dataset type not found")

    send_notification(config.NOTIFICATION, "Extracting features...")
    X = fe.transform_data(dataset)

    labelEncoder = LabelEncoder()
    y = labelEncoder.transform(dataset["type"])

    minMax = MinMaxScaler()
    minMax.fit(X, y)
    XTransform = minMax.transform(X)
    filename_minmax_variants = f"minmax_scaler{algorithm}_{dataset_type}_{test_size}.pkl"
    dump(minMax, open(os.path.join(os.getcwd(), "model", filename_minmax_variants), "wb"))

    send_notification(config.NOTIFICATION, f"Splitting dataset with test size: {test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        XTransform, y, test_size=test_size, random_state=42
    )

    send_notification(config.NOTIFICATION, "Training model...")
    if algorithm == "svm":
        svm = SVM()
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        send_classification_report(config, y_test, y_pred)

        send_notification(config.NOTIFICATION, "Saving model...")
        filename_model_variants = f"model_{algorithm}_{dataset_type}_{test_size}.pkl"
        path_model = os.path.join(os.getcwd(), "model", filename_model_variants)
        svm.save_model(path_model)
    else:
        psosvm = PSOSVM(config)
        psosvm.setData(X_train, y_train, val_size=0.2)
        psosvm.setFitnessFunction(fitness)
        psosvm.train()
        y_pred = psosvm.predict(X_test)
        send_classification_report(config, y_test, y_pred)

        send_notification(config.NOTIFICATION, "Saving model...")
        filename_model_variants = f"model_{algorithm}_{dataset_type}_{test_size}.pkl"
        path_model = os.path.join(os.getcwd(), "model", filename_model_variants)
        psosvm.save_best(path_model)

        path_history = os.path.join(os.getcwd(), "history")
        if not os.path.exists(path_history):
            os.makedirs(path_history)
        filename_history_variants = f"history_{algorithm}_{dataset_type}_{test_size}.csv"
        psosvm.save_history(os.path.join(path_history, filename_history_variants))
    return evaluation(y_test, y_pred)


def training_model(args):
    test_size = args.test_size
    if test_size > 1:
        test_size = test_size / 100
    train(
        test_size=test_size,
        dataset_type=args.dataset,
        algorithm=args.alg,
        config=args.config
    )


def training_all_scenario(args):
    scenario_folder = os.path.join(os.getcwd(), "train_scenario")

    if args.scenario is None:
        scenarios = [f for f in os.listdir(scenario_folder) if f.endswith(".json")]
    else:
        scenarios = [f for f in os.listdir(scenario_folder) if f.endswith(".json") and
                     (args.scenario is not None and f.startswith(args.scenario))]
    scenario_result = []
    for fileName in scenarios:
        with open(os.path.join(scenario_folder, fileName)) as f:
            scenarioType = json.load(f)
            for scenario in scenarioType:
                send_notification(args.config.NOTIFICATION,
                                  f"Training scenario {scenario['name']} started")
                timeStart = datetime.now()
                accuracy, tpr, fpr, fdr = train(
                    test_size=scenario["test_size"],
                    dataset_type=scenario["dataset_type"],
                    algorithm=scenario["algorithm"],
                    fitness=scenario["fitness"],
                    config=args.config
                )
                timeEnd = datetime.now()
                send_notification(args.config.NOTIFICATION,
                                  f"Training scenario {scenario['name']} finished in {timeEnd - timeStart} seconds")
                send_model(args.config.NOTIFICATION, scenario["algorithm"], scenario["dataset_type"], scenario["test_size"])
                scenario_result.append({
                    "name": scenario["name"],
                    "accuracy": accuracy,
                    "tpr": tpr,
                    "fpr": fpr,
                    "fdr": fdr
                })

    # result_scenario_path
    result_scenario_path = os.path.join(os.getcwd(), "result_scenario")
    if not os.path.exists(result_scenario_path):
        os.makedirs(result_scenario_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(os.path.join(result_scenario_path, f"result_scenario_{timestamp}.json"), "w") as f:
        json.dump(scenario_result, f, )
    send_notification(args.config.NOTIFICATION, "Training all scenario is done")
