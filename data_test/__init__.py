import time
import pandas as pd
import datasets
import feature_extraction as fe
from sklearn.model_selection import train_test_split
from model import load_model
from util.LabelEncoder import LabelEncoder
from signature import detect_signature, add_signature, remove_database
import data_evaluation as de
import matplotlib.pyplot as plt
import os
from util.Notification import send_notification


class Argument:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def test_new(args):
    variant = args.variant
    config = args.config
    print(variant)
    test_size = float(variant.split("_")[2])
    dataset_type = variant.split("_")[1]

    if dataset_type == "ecml":
        dataset = datasets.getFormattedECMLDatasets()
    elif dataset_type == "csic":
        dataset = datasets.getFormattedCSICDatasets()
    else:
        raise ValueError("Dataset type not found")

    send_notification(config.NOTIFICATION, f"Testing with variant: {variant}")

    X = fe.transform_data(dataset)
    send_notification(config.NOTIFICATION, "Extracting features...")
    labelEncoder = LabelEncoder()
    y = labelEncoder.transform(dataset["type"])
    model, minMax = load_model(variant)
    XTransform = minMax.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        XTransform, y, test_size=test_size, random_state=42
    )
    y_pred = model.predict(X_test)
    de.send_classification_report(config, y_test, y_pred)
    accuracy, tpr, fpr, fdr = de.evaluation(y_test, y_pred)
    return accuracy, tpr, fpr, fdr, variant


def test_all(args):
    path_model = os.path.join(os.getcwd(), "model")
    result = []
    for f in os.listdir(path_model):
        if f.startswith("model_") and f.endswith(".pkl"):
            argument = {
                "variant": f.replace(".pkl", "").replace("model_", ""),
                "config": args.config,
            }

            result_test = test_new(Argument(argument))
            result.append(result_test)

    # save result to csv
    result = pd.DataFrame(result, columns=["accuracy", "tpr", "fpr", "fdr", "variant"])
    path_result = os.path.join(os.getcwd(), "result")
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    filename_result = f"result_all_test.csv"
    result.to_csv(os.path.join(path_result, filename_result), index=False)

    # Search for the best model Accuracy
    best_accuracy = result.loc[result["accuracy"].idxmax()]
    send_notification(args.config.NOTIFICATION,
                      f"Best Accuracy: {best_accuracy['accuracy']} with variant: {best_accuracy['variant']}")
    # Search for the best model TPR
    best_tpr = result.loc[result["tpr"].idxmax()]
    send_notification(args.config.NOTIFICATION, f"Best TPR: {best_tpr['tpr']} with variant: {best_tpr['variant']}")
    # Search for the best model FPR
    best_fpr = result.loc[result["fpr"].idxmin()]
    send_notification(args.config.NOTIFICATION, f"Best FPR: {best_fpr['fpr']} with variant: {best_fpr['variant']}")
    # Search for the best model FDR
    best_fdr = result.loc[result["fdr"].idxmin()]
    send_notification(args.config.NOTIFICATION, f"Best FDR: {best_fdr['fdr']} with variant: {best_fdr['variant']}")
