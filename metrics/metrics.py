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

    def __init__(self, threshold=0.5, max_auc_pixels=200000, auc_seed=2026):
        self.threshold = threshold
        self.max_auc_pixels = max_auc_pixels
        self.auc_seed = auc_seed
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self._rng = np.random.default_rng(self.auc_seed)
        self._auc_seen = 0
        self._auc_count = 0
        if self.max_auc_pixels is None:
            self._auc_scores = []
            self._auc_targets = []
        else:
            capacity = max(0, int(self.max_auc_pixels))
            self._auc_scores = np.empty(capacity, dtype=np.float32)
            self._auc_targets = np.empty(capacity, dtype=np.uint8)

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
            self._auc_seen += int(pred_probs.size)
            return

        pred_probs = pred_probs.astype(np.float32, copy=False)
        targets = targets.astype(np.uint8, copy=False)

        if self.max_auc_pixels is None:
            self._auc_scores.append(pred_probs.copy())
            self._auc_targets.append(targets.copy())
            self._auc_count += int(pred_probs.size)
            self._auc_seen += int(pred_probs.size)
            return

        capacity = int(self._auc_scores.shape[0])
        num_pixels = int(pred_probs.size)
        if capacity == 0 or num_pixels == 0:
            self._auc_seen += num_pixels
            return

        fill_count = min(capacity - self._auc_count, num_pixels)
        if fill_count > 0:
            start = self._auc_count
            end = start + fill_count
            self._auc_scores[start:end] = pred_probs[:fill_count]
            self._auc_targets[start:end] = targets[:fill_count]
            self._auc_count = end

        remaining = num_pixels - fill_count
        if remaining > 0:
            start = fill_count
            item_numbers = np.arange(
                self._auc_seen + start + 1,
                self._auc_seen + num_pixels + 1,
                dtype=np.float64,
            )
            keep_prob = capacity / item_numbers
            keep_mask = self._rng.random(remaining) < keep_prob
            if np.any(keep_mask):
                replacement_indices = self._rng.integers(
                    0,
                    capacity,
                    size=int(keep_mask.sum()),
                    dtype=np.int64,
                )
                self._auc_scores[replacement_indices] = pred_probs[start:][keep_mask]
                self._auc_targets[replacement_indices] = targets[start:][keep_mask]

        self._auc_seen += num_pixels

    @staticmethod
    def _flatten_samples(samples, dtype):
        if isinstance(samples, list):
            if not samples:
                return np.empty(0, dtype=dtype)
            return np.concatenate([np.asarray(item, dtype=dtype).reshape(-1) for item in samples], axis=0)
        return np.asarray(samples, dtype=dtype).reshape(-1)

    def state_dict(self):
        if self.max_auc_pixels is None:
            auc_scores = [item.copy() for item in self._auc_scores]
            auc_targets = [item.copy() for item in self._auc_targets]
        else:
            auc_scores = self._auc_scores[: self._auc_count].copy()
            auc_targets = self._auc_targets[: self._auc_count].copy()

        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "auc_scores": auc_scores,
            "auc_targets": auc_targets,
            "auc_count": self._auc_count,
            "auc_seen": self._auc_seen,
        }

    def load_state_dict(self, state):
        self.tp = int(state["tp"])
        self.fp = int(state["fp"])
        self.tn = int(state["tn"])
        self.fn = int(state["fn"])
        scores = self._flatten_samples(state.get("auc_scores", []), np.float32)
        targets = self._flatten_samples(state.get("auc_targets", []), np.uint8)
        self._auc_seen = int(state.get("auc_seen", state.get("auc_count", scores.size)))

        if self.max_auc_pixels is None:
            self._auc_scores = [scores] if scores.size > 0 else []
            self._auc_targets = [targets] if targets.size > 0 else []
            self._auc_count = int(scores.size)
            return

        capacity = int(self._auc_scores.shape[0])
        keep = min(capacity, int(scores.size))
        self._auc_count = keep
        if keep > 0:
            self._auc_scores[:keep] = scores[:keep]
            self._auc_targets[:keep] = targets[:keep]

    def merge_state_dict(self, state):
        self.tp += int(state["tp"])
        self.fp += int(state["fp"])
        self.tn += int(state["tn"])
        self.fn += int(state["fn"])
        scores = self._flatten_samples(state.get("auc_scores", []), np.float32)
        targets = self._flatten_samples(state.get("auc_targets", []), np.uint8)
        if scores.size == 0:
            return

        if self.max_auc_pixels is None:
            self._auc_scores.append(scores)
            self._auc_targets.append(targets)
            self._auc_count += int(scores.size)
            self._auc_seen += int(state.get("auc_seen", scores.size))
        else:
            self._store_auc_samples(scores, targets)

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
        if self._auc_count == 0:
            return 0.0

        if self.max_auc_pixels is None:
            scores = np.concatenate(self._auc_scores, axis=0)
            targets = np.concatenate(self._auc_targets, axis=0)
        else:
            scores = self._auc_scores[: self._auc_count]
            targets = self._auc_targets[: self._auc_count]
        try:
            return _roc_auc_score(targets, scores)
        except ValueError:
            return 0.0
