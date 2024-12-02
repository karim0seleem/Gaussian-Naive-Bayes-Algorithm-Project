import numpy as np
from tabulate import tabulate

def Evaluate(y_test, y_pred, class_labels=None):
    """
    Evaluates the performance of a classifier using the given test data and predictions.

    Parameters
    ----------
    y_test : array-like
        The true class labels for the test data.
    y_pred : array-like
        The predicted class labels for the test data.
    class_labels : list or None, optional
        List of class labels. If provided, it will be used to label the rows and columns of the confusion matrix.

    Returns
    -------
    None
    """
    # Convert y_test and y_pred to numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Number of true positives, false positives, true negatives, and false negatives
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    # Handle division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # Handle F1 score calculation
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    # Confusion matrix
    cm = np.array([[tp, fp], [fn, tn]])
    
    # Prepare data for tabulate
    if class_labels is None:
        class_labels = ["Positive", "Negative"]
    table_data = [[class_labels[0], cm[0, 0], cm[0, 1]],
                  [class_labels[1], cm[1, 0], cm[1, 1]]]
    
    # Print evaluation results
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Confusion Matrix:")
    print(tabulate(table_data, headers=["", class_labels[0], class_labels[1]], tablefmt="grid"))
