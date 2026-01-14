import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

def compute_metrics(p):
    """
    Computes and returns a dictionary of metrics for the evaluation.

    Args:
        p (EvalPrediction): A named tuple with `predictions` and `label_ids` fields.

    Returns:
        dict: A dictionary containing the F1 score, precision, and recall.
    """
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='weighted'
    )
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
