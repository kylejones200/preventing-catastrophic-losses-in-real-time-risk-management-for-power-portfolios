//! Historical Value-at-Risk (single quantile, sorted returns).

pub fn historical_var(returns: &[f64], alpha: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64) * alpha).floor() as usize;
    -sorted[idx.min(sorted.len() - 1)]
}
