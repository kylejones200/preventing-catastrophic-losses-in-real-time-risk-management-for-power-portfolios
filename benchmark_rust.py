#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import historical_var  # noqa: E402

def main() -> None:
    r = np.ascontiguousarray(np.sin(np.arange(2000) * 0.01) * 0.02)
    alpha = 0.05
    t0 = time.perf_counter()
    for _ in range(200):
        historical_var(r, alpha)
    py_s = time.perf_counter() - t0
    try:
        import preventing_catastrophic_losses_in_real_time_risk_management_for_power_portfolios_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(r, alpha, 10000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        historical_var(r, alpha),
        rs.historical_var_py(r, alpha),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
