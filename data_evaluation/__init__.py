from sklearn.metrics import confusion_matrix
from util.Notification import send_notification


def evaluation(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    new_cf = [[cf[1][1], cf[1][0]], [cf[0][1], cf[0][0]]]
    accuracy = __calculate_accuracy(new_cf)
    tpr = __calculate_TruePositiveRate(new_cf)
    fpr = __calculate_FalsePositiveRate(new_cf)
    fdr = __calculate_FalseDiscoveryRate(new_cf)
    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "fdr": fdr
    }


def classification_report(y_true, y_pred):
    accuracy, tpr, fpr, fdr = evaluation(y_true, y_pred)

    print("Confusion Matrix: ")
    # round 2
    accuracy = round(accuracy, 2)
    tpr = round(tpr, 2)
    fpr = round(fpr, 2)
    fdr = round(fdr, 2)
    # print like classification report
    print("Classification Report: ")
    print("Confusion Matrix: ")
    print("True Positive Rate: ", tpr)
    print("False Positive Rate: ", fpr)
    print("False Discovery Rate: ", fdr)
    print("Accuracy: ", accuracy)


def send_classification_report(config, y_true, y_pred):
    accuracy, tpr, fpr, fdr = evaluation(y_true, y_pred)
    send_notification(config.NOTIFICATION, "Accuracy: " + str(accuracy))
    send_notification(config.NOTIFICATION, "True Positive Rate: " + str(tpr))
    send_notification(config.NOTIFICATION, "False Positive Rate: " + str(fpr))
    send_notification(config.NOTIFICATION, "False Discovery Rate: " + str(fdr))
    classification_report(y_true, y_pred)


def __calculate_accuracy(cf):
    return (cf[0][0] + cf[1][1]) / (cf[0][0] + cf[1][1] + cf[0][1] + cf[1][0])


def __calculate_TruePositiveRate(cf):
    return cf[1][1] / (cf[1][1] + cf[1][0])


def __calculate_FalsePositiveRate(cf):
    return cf[0][1] / (cf[0][1] + cf[0][0])


def __calculate_FalseDiscoveryRate(cf):
    return cf[0][1] / (cf[0][1] + cf[1][1])
