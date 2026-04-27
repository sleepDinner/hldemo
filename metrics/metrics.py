import math

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except ImportError:
    _sklearn_roc_auc_score = None


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else 0.0


def _mcc_denominator(tp, fp, tn, fn):
    terms = [
        float(tp + fp),
        float(tp + fn),
        float(tn + fp),
        float(tn + fn),
    ]
    if any(term <= 0.0 for term in terms):
        return 0.0
    return math.sqrt(terms[0] * terms[1] * terms[2] * terms[3])


def _roc_auc_score(targets, scores):
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(targets, scores))

    targets = np.asarray(targets).astype(np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = targets == 1
    negatives = targets == 0
    num_pos = int(positives.sum())
    num_neg = int(negatives.sum())
    if num_pos == 0 or num_neg == 0:
        raise ValueError("AUC requires both positive and negative samples.")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = ranks[positives].sum()
    auc = (rank_sum_pos - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def binary_metrics(pred_probs, targets, threshold=0.5):
    tracker = RunningBinaryMetrics(threshold=threshold, max_auc_pixels=None)
    tracker.update(pred_probs, targets)
    return tracker.compute()


class RunningBinaryMetrics:
    """Streaming pixel-level metrics for binary tamper masks."""

    def __init__(self, threshold=0.5, max_auc_pixels=200000):
        self.threshold = threshold
        self.max_auc_pixels = max_auc_pixels
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self._auc_scores = []
        self._auc_targets = []
        self._auc_count = 0

    def update(self, pred_probs, targets):
        pred_probs = self._to_numpy(pred_probs).reshape(-1)
        targets = self._to_numpy(targets).reshape(-1).astype(np.uint8)
        preds = (pred_probs >= self.threshold).astype(np.uint8)

        self.tp += int(((preds == 1) & (targets == 1)).sum())
        self.fp += int(((preds == 1) & (targets == 0)).sum())
        self.tn += int(((preds == 0) & (targets == 0)).sum())
        self.fn += int(((preds == 0) & (targets == 1)).sum())
        self._store_auc_samples(pred_probs, targets)

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _store_auc_samples(self, pred_probs, targets):
        if self.max_auc_pixels == 0:
            return

        if self.max_auc_pixels is None:
            keep = pred_probs.size
        else:
            remaining = self.max_auc_pixels - self._auc_count
            if remaining <= 0:
                return
            keep = min(remaining, pred_probs.size)

        if keep < pred_probs.size:
            indices = np.linspace(0, pred_probs.size - 1, num=keep, dtype=np.int64)
            pred_probs = pred_probs[indices]
            targets = targets[indices]

        self._auc_scores.append(pred_probs.astype(np.float32))
        self._auc_targets.append(targets.astype(np.uint8))
        self._auc_count += int(keep)

    def state_dict(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "auc_scores": self._auc_scores,
            "auc_targets": self._auc_targets,
            "auc_count": self._auc_count,
        }

    def load_state_dict(self, state):
        self.tp = int(state["tp"])
        self.fp = int(state["fp"])
        self.tn = int(state["tn"])
        self.fn = int(state["fn"])
        self._auc_scores = list(state.get("auc_scores", []))
        self._auc_targets = list(state.get("auc_targets", []))
        self._auc_count = int(state.get("auc_count", 0))

    def merge_state_dict(self, state):
        self.tp += int(state["tp"])
        self.fp += int(state["fp"])
        self.tn += int(state["tn"])
        self.fn += int(state["fn"])
        self._auc_scores.extend(state.get("auc_scores", []))
        self._auc_targets.extend(state.get("auc_targets", []))
        self._auc_count += int(state.get("auc_count", 0))

    def compute(self):
        precision = _safe_divide(self.tp, self.tp + self.fp)
        recall = _safe_divide(self.tp, self.tp + self.fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        iou = _safe_divide(self.tp, self.tp + self.fp + self.fn)
        fpr = _safe_divide(self.fp, self.fp + self.tn)
        mcc_den = _mcc_denominator(self.tp, self.fp, self.tn, self.fn)
        mcc = _safe_divide(self.tp * self.tn - self.fp * self.fn, mcc_den)
        auc = self._compute_auc()

        return {
            "f1": f1,
            "iou": iou,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "mcc": mcc,
            "fpr": fpr,
        }

    def _compute_auc(self):
        if not self._auc_scores:
            return 0.0

        scores = np.concatenate(self._auc_scores, axis=0)
        targets = np.concatenate(self._auc_targets, axis=0)
        try:
            return _roc_auc_score(targets, scores)
        except ValueError:
            return 0.0
