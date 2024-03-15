"""
Main script to execute the machine learning workflow.
"""
import warnings
import pandas as pd

from IPython.display import display
from sklearn.model_selection import train_test_split

from load_data import load_data
from classifiers.random_forest_classifier import random_forest_classifier_scores
from classifiers.dummy_classifier import evaluate_stratified_dummy_classifier
from classifiers.dummy_classifier import evaluate_most_frequent_dummy_classifier
from classifiers.svc_classifier import svc_classification_score
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier_grid_search

from visualization import classifier_performance_bar_chart

# Filter specific warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Load data
X, y = load_data("assets/spam.csv")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# RandomForestClassifier
acc_rf, prec_rf, rec_rf, _ = random_forest_classifier_scores(
    X_train, X_test, y_train, y_test)

# Dummy classifiers
acc_strat, prec_strat, rec_strat, _ = evaluate_stratified_dummy_classifier(
    X_train, X_test, y_train, y_test)
acc_mostfreq, prec_mostfreq, rec_mostfreq, _ = evaluate_most_frequent_dummy_classifier(
    X_train, X_test, y_train, y_test)

# SVC Classifier
acc_svc, prec_svc, rec_svc, conf_matrix_svc = svc_classification_score(
    X_train, X_test, y_train, y_test)

# Logistic Regression Classifier
acc_lrs, prec_lrs, rec_lrs, conf_matrix_lrs = logistic_regression_classifier(
                                                X_train, X_test, y_train, y_test)

# Logistic Regression Classifier with Grid Search
acc_lrsgs, prec_lrsgs, rec_lrsgs, _ = logistic_regression_classifier_grid_search(
    X_train, X_test, y_train, y_test)
# Create a dictionary to hold the results
results = {
    'Classifier': ['Random Forest', 'Stratified Dummy', 'Most Frequent Dummy', 'SVC',
                   'Logistic Regression', 'Logistic Regression with Grid Search'],
    'Accuracy': [acc_rf, acc_strat, acc_mostfreq, acc_svc,acc_lrs, acc_lrsgs],
    'Precision': [prec_rf, prec_strat, prec_mostfreq, prec_svc,prec_lrs, prec_lrsgs],
    'Recall': [rec_rf, rec_strat, rec_mostfreq, rec_svc,rec_lrs, rec_lrsgs]
}
results_df = pd.DataFrame(results)
display(results_df)

# Plot Performance Comparison of Classifiers
classifier_performance_bar_chart(results_df)
