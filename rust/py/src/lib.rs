use preventing_catastrophic_losses_in_real_time_risk_management_for_power_portfolios_core::historical_var;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn historical_var_py(returns: PyReadonlyArray1<f64>, alpha: f64) -> PyResult<f64> {
    Ok(historical_var(returns.as_slice()?, alpha))
}

#[pyfunction]
#[pyo3(signature = (returns, alpha, iterations=10_000))]
fn bench_kernel_py(
    returns: PyReadonlyArray1<f64>,
    alpha: f64,
    iterations: usize,
) -> PyResult<f64> {
    let r = returns.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = historical_var(&r, alpha);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn preventing_catastrophic_losses_in_real_time_risk_management_for_power_portfolios_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(historical_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
