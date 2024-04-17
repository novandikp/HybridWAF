import time
import pandas as pd
import datasets
import feature_extraction as fe
from sklearn.model_selection import train_test_split
from model import minMax, model
from util.LabelEncoder import LabelEncoder
from signature import detect_signature, add_signature, remove_database
import data_evaluation as de
import matplotlib.pyplot as plt


def test(args):
    # check
    test_size = 0.25
    remove_database()
    dataset = datasets.getFormattedECMLDatasets()

    labelEncoder = LabelEncoder()
    X = fe.transform_data_with_time(dataset)
    y = labelEncoder.transform(dataset["type"])

    timeX = X["time"]
    X = X.drop(columns=["time"])
    XTransform = minMax.transform(X)
    # add time to XTransform
    XTransform = pd.DataFrame(XTransform, columns=X.columns)
    XTransform["time"] = timeX
    X_train, X_test, y_train, y_test = train_test_split(
        XTransform, y, test_size=test_size, random_state=27
    )

    # print train and test size
    print("Train size: ", X_train.shape)
    print("Test size: ", X_test.shape)
    test_time = X_test["time"]
    X_test = X_test.drop(columns=["time"])
    result_prediction = []
    real_condition = []
    time_predict = []
    model_used = []
    # iterate 2 times
    for i in range(2):
        print(i)
        # iterate over x_test and y_test
        for x, y, z in zip(X_test.values, y_test, test_time):
            start_time = time.time()
            resultSignature = detect_signature(x)
            real_condition.append(y)
            if resultSignature is None:
                result = model.predict([x])
                add_signature(x, bool(result[0]))
                end_time = time.time()
                classification_time = end_time - start_time
                classification_time = round((classification_time + z) * 1000, 2)
                result_prediction.append(result[0])
                time_predict.append(classification_time)
                model_used.append("Anomaly Based Detection")
            else:
                end_time = time.time()
                classification_time = end_time - start_time
                classification_time = round((classification_time + z) * 1000, 2)
                result_prediction.append(int(resultSignature))
                time_predict.append(classification_time)
                model_used.append("Signature Based Detection")

        if i == 0:
            de.classification_report(real_condition, result_prediction)

    # save history to csv
    result = pd.DataFrame(
        {
            "real_condition": real_condition,
            "result_prediction": result_prediction,
            "time_predict": time_predict,
            "model_used": model_used,
        }
    )

    result.to_csv("test_result.csv")
    anomaly_based = result[result["model_used"] == "Anomaly Based Detection"]
    signature_based = result[result["model_used"] == "Signature Based Detection"]

    # reset index
    anomaly_based = anomaly_based.reset_index()
    signature_based = signature_based.reset_index()

    anomaly_based = anomaly_based.head(20)
    signature_based = signature_based.head(20)

    plt.plot(anomaly_based.index, anomaly_based["time_predict"], label="Anomaly Based Detection")
    plt.plot(signature_based.index, signature_based["time_predict"], label="Signature Based Detection")
    plt.xlabel("Index")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()
