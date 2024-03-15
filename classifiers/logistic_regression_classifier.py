"""
Module to define functions for evaluating Logistic Regression Classifier.
"""
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from evaluate import precision_recall_curve_wrapper



def logistic_regression_classifier(x_train, x_test, y_train, y_test):
    """
    Evaluate recall and True Positive rate.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        conf_matrix (array): Confusion matrix.
    """
    accuracy, precision, recall, conf_matrix = None, None, None, None
    logreg_classifier = LogisticRegression()
    logreg_classifier.fit(x_train, y_train)
    prob_spam = logreg_classifier.predict_proba(x_test)[:, 1]
    precision, recall =  precision_recall_curve_wrapper(y_test, prob_spam)
    recall = recall[np.argmax(precision >= 0.90)]
    y_pred = (prob_spam >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, conf_matrix


def logistic_regression_classifier_grid_search(x_train, x_test, y_train, y_test):
    """
    Perform grid search with logistic regression classifier.

    Parameters:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        conf_matrix (array): Confusion matrix.
        mean_test_scores (array): Mean test scores from grid search.
    """
    accuracy = precision = recall = conf_matrix = None
    # Define parameter grid for grid search
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.005, 0.01, 0.05, 0.1, 1, 10]
    }
    # Create logistic regression classifier
    logreg_classifier = LogisticRegression(solver='liblinear', random_state=42)
    grid_search = GridSearchCV(logreg_classifier, param_grid, scoring='precision', cv=5)
    grid_search.fit(x_train, y_train)
    # Evaluate the best model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, conf_matrix
