"""
Module for visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def GridSearch_Heatmap(scores):
    """
    Generate heatmap for grid search results.

    Parameters:
        scores (array): Grid search results.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.yticks(rotation=0)

    sns.heatmap(
        scores,
        xticklabels=["l1", "l2"],
        yticklabels=[0.005, 0.01, 0.05, 0.1, 1, 10],
    )
    plt.xlabel('Penalty')
    plt.ylabel('C')

    plt.show()
