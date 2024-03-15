# Spam Email Detection - evaluate different M/L classifiers

## Overview
This project aims to evaluate different machine learning classifiers for the task of spam email detection. Spam email detection is a critical application in email filtering systems, where the objective is to automatically identify unsolicited, mass-produced messages containing irrelevant or inappropriate content from legitimate emails.

## Dataset
The dataset used in this project contains labeled email samples, sourced from [insert source link]. Each sample is classified as either "spam" (Class == 1) or "not spam" (Class == 0). The dataset will be split into training and testing sets to train and evaluate the machine learning models.

## Classifiers Implemented
The following classifiers are evaluated in this project:

- Random Forest Classifier: Ensemble learning method that constructs multiple decision trees during training.
- Dummy Classifiers: Baseline models for comparison, including stratified and most frequent strategies.
- Support Vector Classifier (SVC): Powerful classifier aiming to find the hyperplane that best separates different classes in the feature space.
- Logistic Regression Classifier: Simple yet effective linear model used for binary classification tasks.

## Evaluation Metrics
The performance of the classifiers is assessed using the following evaluation metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix

## Visualization

The results of the model evaluationare plotted to provide insights into the performance and comparison of the models.

### Metrics Evaluation

![Metrics Evaluation](assets\classifier_performance.png)


## Usage
To run the project:

1. Clone this repository to your local machine.

    git clone https://github.com/your-username/machine-learning-workflow.git

2. Install the required dependencies listed in requirements.txt.
    
    pip install -r requirements.txt

3. Execute the main script main.py to perform data loading, preprocessing, model training, evaluation, and visualization.

    python main.py

## Contributors
- Vinoj Bethelli


