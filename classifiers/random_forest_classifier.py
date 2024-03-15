"""
Module to define functions for evaluating Random Forest Classifier.
"""

from sklearn.ensemble import RandomForestClassifier
from pipeline import create_pipeline
from evaluate import evaluate_model

def random_forest_classifier_scores(x_train, x_test, y_train, y_test):
    """
    Evaluate the RandomForestClassifier

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        result_tuple (tuple): Results of the stratified dummy classifier.
    """
    model = create_pipeline(RandomForestClassifier())
    return evaluate_model(model, x_train, x_test, y_train, y_test)
