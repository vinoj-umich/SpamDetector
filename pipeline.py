"""
Module to create a preprocessing pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(model):
    """
    Create a preprocessing pipeline with a specified model.

    Parameters:
        model: Estimator object.

    Returns:
        pipeline: Preprocessing pipeline.
    """
    preprocessing_steps = [('scaler', StandardScaler())]
    steps = preprocessing_steps + [('classifier', model)]
    pipeline = Pipeline(steps)
    return pipeline
