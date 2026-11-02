"""Explicit metric definitions for rare residue-pair contacts.

Ranking uses logits; sigmoid values are diagnostic scores, not calibrated
probabilities. Top-K metrics average uniformly within a tie at the cutoff.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score


def checked_arrays(y, score):
    y, score = np.asarray(y).ravel(), np.asarray(score, dtype=np.float64).ravel()
    if not len(y) or y.shape != score.shape:
        raise ValueError("Labels and scores must have matching nonempty shapes")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Expected binary labels and finite scores")
    if not (0 < y.sum() < len(y)):
        raise ValueError("Ranking evaluation requires both label classes")
    return y.astype(np.uint8), score


def independent_auc(y, score):
    """Mann-Whitney/rank-sum definition; tied pairs receive half credit."""
    y, score = checked_arrays(y, score)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - positive*(positive+1)/2) / (positive*negative))


def independent_ap(y, score):
    """Sum precision * recall increment at distinct score thresholds."""
    y, score = checked_arrays(y, score)
    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[order], score[order]
    end = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y)-1]
    tp = np.cumsum(sorted_y, dtype=np.int64)[end]
    increment = np.diff(np.r_[0, tp]) / int(y.sum())
    precision = tp / (end+1)
    return float(np.dot(increment, precision))


def topk_expected(y, score, k):
    y, score = checked_arrays(y, score)
    k = min(max(1, int(k)), len(y))
    cutoff = np.partition(score, len(score)-k)[len(score)-k]
    above, tied = score > cutoff, score == cutoff
    slots = k-int(above.sum())
    expected_tp = float(y[above].sum()) + slots*float(y[tied].mean())
    precision = expected_tp/k
    return {"k": k, "expected_tp": expected_tp, "precision": precision,
            "recall": expected_tp/int(y.sum()), "enrichment": precision/float(y.mean()),
            "cutoff_tie_size": int(tied.sum()), "cutoff_tie_slots": slots}


def threshold_metrics(y, score, threshold=0.5):
    y, score = checked_arrays(y, score)
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall, specificity = tp/(tp+fn), tn/(tn+fp)
    denominator = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_positive": tp+fp, "precision": precision, "recall": recall,
            "specificity": specificity, "false_positive_rate": 1-specificity,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "mcc": (tp*tn-fp*fn)/denominator if denominator else 0.0,
            "balanced_accuracy": (recall+specificity)/2,
            "accuracy": (tp+tn)/len(y), "precision_defined": bool(tp+fp)}


def ranking_metrics(y, logits):
    y, logits = checked_arrays(y, logits)
    precision, recall, _ = precision_recall_curve(y, logits)
    return {"roc_auc": float(roc_auc_score(y, logits)),
            "average_precision": float(average_precision_score(y, logits)),
            "pr_auc_trapezoid": float(auc(recall, precision)),
            "standardized_partial_roc_auc_fpr_0_01": float(roc_auc_score(y, logits, max_fpr=0.01)),
            "n_pairs": len(y), "n_contacts": int(y.sum()), "prevalence": float(y.mean())}


def uncalibrated_score_diagnostics(y, logits, scores):
    y, logits = checked_arrays(y, logits)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.shape != y.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected one finite sigmoid score in [0,1] per label")
    return {"brier_uncalibrated_score": float(np.mean((scores-y)**2)),
            "log_loss_uncalibrated_score": float(np.mean(np.logaddexp(0, logits)-y*logits)),
            "mean_uncalibrated_score": float(scores.mean())}
