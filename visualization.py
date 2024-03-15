"""
Module for visualization functions.
"""
import numpy as np
import matplotlib.pyplot as plt

def classifier_performance_bar_chart(results_df):
    """
    Visulaization to show the performce of different classifiers in the form of bar chart
    """
    plt.style.use('ggplot')
    bar_width = 0.35
    positions = np.arange(len(results_df))
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Plot Accuracy
    axs[0].barh(positions, results_df['Accuracy'], height=bar_width,
                color='skyblue', label='Accuracy')
    axs[0].set_xlabel('Accuracy')
    axs[0].set_yticks(positions)
    axs[0].set_yticklabels(results_df['Classifier'])
    axs[0].legend()
    # Plot Precision
    axs[1].barh(positions, results_df['Precision'], height=bar_width,
                color='lightgreen', label='Precision')
    axs[1].set_xlabel('Precision')
    axs[1].set_yticks(positions)
    axs[1].set_yticklabels(results_df['Classifier'])
    axs[1].legend()
    # Plot Recall
    axs[2].barh(positions, results_df['Recall'], height=bar_width,
                color='salmon', label='Recall')
    axs[2].set_xlabel('Recall')
    axs[2].set_yticks(positions)
    axs[2].set_yticklabels(results_df['Classifier'])
    axs[2].legend()
    fig.suptitle('Performance Comparison of Classifiers')
    plt.tight_layout()
    plt.show()
