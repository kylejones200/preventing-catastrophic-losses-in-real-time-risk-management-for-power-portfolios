"""Historical Value-at-Risk (single quantile, sorted returns)."""

from __future__ import annotations

import numpy as np


def historical_var(returns: np.ndarray, alpha: float) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    sorted_r = np.sort(r)
    idx = int(np.floor(len(sorted_r) * alpha))
    return float(-sorted_r[min(idx, len(sorted_r) - 1)])
