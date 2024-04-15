import datasets
import feature_extraction as fe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_training.PSOSVM import PSOSVM
from data_training.SVM import SVM
from util.Notification import send_notification
from data_evaluation import send_classification_report
from util.LabelEncoder import LabelEncoder
from cloudpickle import dump
import os


def train(test_size=0.25, dataset_type="ecml", algorithm="psosvm", config=None):
    send_notification(config.NOTIFICATION,
                      f"Training with test size: {test_size}, dataset: {dataset_type}, algorithm: {algorithm}")
    send_notification(config.NOTIFICATION, "Loading dataset...")
    if dataset_type == "ecml":
        dataset = datasets.getFormattedECMLDatasets()
    elif dataset_type == "csic":
        dataset = datasets.getFormattedCSICDatasets()
    else:
        dataset = datasets.getFormattedDatasets()


    send_notification(config.NOTIFICATION, "Extracting features...")
    X = fe.transform_data(dataset)

    labelEncoder = LabelEncoder()
    y = labelEncoder.transform(dataset["type"])

    minMax = MinMaxScaler()
    minMax.fit(X, y)
    XTransform = minMax.transform(X)
    dump(minMax, open(os.path.join(os.getcwd(), "model", "minmax_scaler.pkl"), "wb"))

    send_notification(config.NOTIFICATION, f"Splitting dataset with test size: {test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        XTransform, y, test_size=test_size, random_state=27
    )

    send_notification(config.NOTIFICATION, "Training model...")
    if algorithm == "svm":
        svm = SVM(config)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        send_classification_report(y_test, y_pred)

        send_notification(config.NOTIFICATION, "Saving model...")
        filename_model_variants = f"model_{algorithm}_{dataset_type}_{test_size}.pkl"
        path_model = os.path.join(os.getcwd(), "model", filename_model_variants)
        svm.save_model(path_model)
    else:
        psosvm = PSOSVM(config)
        psosvm.setData(X_train, y_train, val_size=0.2, random_state=1)
        psosvm.train()
        y_pred = psosvm.predict(X_test)
        send_classification_report(config, y_test, y_pred)

        send_notification(config.NOTIFICATION, "Saving model...")
        filename_model_variants = f"model_{algorithm}_{dataset_type}_{test_size}.pkl"
        path_model = os.path.join(os.getcwd(), "model", filename_model_variants)
        psosvm.save_best(path_model)


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
