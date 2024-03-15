"""
Module to evaluate model performance.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix,precision_recall_curve, roc_curve



def evaluate_model(pipeline, x_train, x_test, y_train, y_test):
    """
    Evaluate model performance on test data.

    Parameters:
        pipeline: Preprocessing pipeline.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        conf_matrix (array): Confusion matrix.
    """
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, conf_matrix

def precision_recall_curve_wrapper(y_test, prob_spam):
    """
    a wrappep precision & recall curve
    """
    precision, recall, _ = precision_recall_curve(y_test, prob_spam)
    return precision, recall

def roc_curve_wrapper(y_test, prob_spam):
    """
    a wrappep roc curve
    """
    fpr, tpr, _ = roc_curve(y_test, prob_spam)
    return fpr, tpr
