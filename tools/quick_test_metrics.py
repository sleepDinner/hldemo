import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from metrics import RunningBinaryMetrics, binary_metrics


def main():
    probs = np.array([0.1, 0.8, 0.4, 0.9], dtype=np.float32)
    targets = np.array([0, 1, 0, 1], dtype=np.uint8)
    print(binary_metrics(probs, targets))

    tracker = RunningBinaryMetrics(threshold=0.5, max_auc_pixels=16)
    tracker.tp = 10**8
    tracker.fp = 10**8
    tracker.tn = 10**8
    tracker.fn = 10**8
    metrics = tracker.compute()
    assert "mcc" in metrics
    assert np.isfinite(metrics["mcc"])
    print("quick metrics test passed")


if __name__ == "__main__":
    main()
