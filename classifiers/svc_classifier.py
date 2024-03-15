"""
Module to define functions for evaluating SVC Classifier.
"""

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from evaluate import evaluate_model


def svc_classification_score(x_train, x_test, y_train, y_test):
    """
    Evaluate the SVC classifier.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        result_tuple (tuple): Results of the most frequent dummy classifier.
    """
    svc_classifier = SVC()
    svc_classifier.fit(x_train, y_train)
    # Evaluate the pipeline on the test data
    return evaluate_model(svc_classifier, x_train, x_test, y_train, y_test)



def svc_classification_decision_score(x_train, x_test, y_train, y_test):
    """
    Evaluate the SVC classifier with decision function.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        result_tuple (tuple): Results of the most frequent dummy classifier.
    """
    conf_mtrx = None
    svc_classifier = SVC(C=1e9, gamma=1e-8)
    svc_classifier.fit(x_train, y_train)
    accuracy, precision, recall, _= evaluate_model(svc_classifier, x_train, x_test, y_train, y_test)
    decision_scores = svc_classifier.decision_function(x_test)
    y_pred = (decision_scores > -100).astype(int)
    conf_mtrx = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, conf_mtrx
