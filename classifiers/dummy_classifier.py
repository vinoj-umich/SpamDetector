"""
Module to define functions for evaluating dummy classifiers.
"""

from sklearn.dummy import DummyClassifier
from evaluate import evaluate_model

def evaluate_stratified_dummy_classifier(x_train, x_test, y_train, y_test):
    """
    Evaluate the stratified dummy classifier.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        result_tuple (tuple): Results of the stratified dummy classifier.
    """
    dummy_classifier = DummyClassifier(strategy="stratified", random_state=0)
    dummy_classifier.fit(x_train, y_train)
    return evaluate_model(dummy_classifier, x_train, x_test, y_train, y_test)


def evaluate_most_frequent_dummy_classifier(x_train, x_test, y_train, y_test):
    """
    Evaluate the most frequent dummy classifier.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        result_tuple (tuple): Results of the most frequent dummy classifier.
    """
    dummy_classifier = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_classifier.fit(x_train, y_train)
    return evaluate_model(dummy_classifier, x_train, x_test, y_train, y_test)
    