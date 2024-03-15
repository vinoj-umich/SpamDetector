"""
Module to define functions for evaluating Logistic Regression Classifier.
"""
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from evaluate import precision_recall_curve_wrapper, roc_curve_wrapper


def logistic_regression_classifier(x_train, x_test, y_train, y_test):
    """
    Evaluate recall and True Positive rate.

    Parameters:
        x_train (DataFrame): Training features.
        x_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        rec (float): Recall score.
        tpr (float): True Positive rate.
    """
    rec, tpr = None, None

    logreg_classifier = LogisticRegression()
    logreg_classifier.fit(x_train, y_train)
    prob_spam = logreg_classifier.predict_proba(x_test)[:, 1]

    precision, recall =  precision_recall_curve_wrapper(y_test, prob_spam)
    fpr, tpr = roc_curve_wrapper(y_test, prob_spam)

    rec = recall[np.argmax(precision >= 0.90)]
    tpr = tpr[np.argmax(fpr >= 0.10)]

    return rec, tpr



def logistic_regression_classifier_with_grid_search(X_train, y_train):
    """
    Perform grid search with logistic regression classifier.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        mean_test_scores (array): Mean test scores from grid search.
    """
    mean_test_scores = None

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.005, 0.01, 0.05, 0.1, 1, 10]
    }

    logreg_classifier = LogisticRegression(solver='liblinear', random_state=42)
    grid_search = GridSearchCV(logreg_classifier, param_grid, scoring='precision', cv=5)
    grid_search.fit(X_train, y_train)

    mean_test_scores = np.array(grid_search.cv_results_['mean_test_score']).reshape(6, 2)

    return mean_test_scores
