"""
Main script to execute the machine learning workflow.
"""

from sklearn.model_selection import train_test_split

from load_data import load_data
from classifiers.random_forest_classifier import random_forest_classifier_scores 
from classifiers.dummy_classifier import evaluate_stratified_dummy_classifier, evaluate_most_frequent_dummy_classifier
from classifiers.svc_classifier import svc_classification_decision_score,svc_classification_score
from classifiers.logistic_regression_classifier import logistic_regression_classifier, logistic_regression_classifier_with_grid_search
from visualization import GridSearch_Heatmap

from evaluate import evaluate_model

# Load data
X, y = load_data("assets/spam.csv")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# RandomForestClassifier
accuracy_rf, precision_rf, recall_rf, conf_matrix_rf = random_forest_classifier_scores(X_train, X_test, y_train, y_test)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("Confusion Matrix:")
print(conf_matrix_rf)

# Dummy classifiers
accuracy_dummy_stratified, precision_dummy_stratified, recall_dummy_stratified, _ = evaluate_stratified_dummy_classifier(X_train, X_test, y_train, y_test)
accuracy_dummy_most_frequent, precision_dummy_most_frequent, recall_dummy_most_frequent, _ = evaluate_most_frequent_dummy_classifier(X_train, X_test, y_train, y_test)

print("\nResults (Dummy Classifiers):")
print("Stratified Dummy Classifier:")
print("Accuracy:", accuracy_dummy_stratified)
print("Precision:", precision_dummy_stratified)
print("Recall:", recall_dummy_stratified)

print("\nMost Frequent Dummy Classifier:")
print("Accuracy:", accuracy_dummy_most_frequent)
print("Precision:", precision_dummy_most_frequent)
print("Recall:", recall_dummy_most_frequent)

# SVC Classifier
accuracy_svc, precision_svc, recall_svc, conf_matrix_svc = svc_classification_score(X_train, X_test, y_train, y_test)
print("\nResults (SVC Classifier):")
print("Accuracy:", accuracy_svc)
print("Precision:", precision_svc)
print("Recall:", recall_svc)
print("Confusion Matrix:")
print(conf_matrix_svc)

#Logistic Regression Classifier with Grid Search
mean_test_scores = logistic_regression_classifier_with_grid_search(X_train, y_train)
print("\Results (Logistic Regression Classifier with Grid Search):")
print("Mean Test Scores:")
print(mean_test_scores)

# Plot heatmap for Task Four
print("\nPlotting Heatmap for Task Four Results (Logistic Regression Classifier with Grid Search):")
GridSearch_Heatmap(mean_test_scores)
