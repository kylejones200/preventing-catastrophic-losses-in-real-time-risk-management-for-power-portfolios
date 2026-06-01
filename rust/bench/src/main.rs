use preventing_catastrophic_losses_in_real_time_risk_management_for_power_portfolios_core::historical_var;

fn main() {
    let r: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.01).sin() * 0.02).collect();
    for _ in 0..10000 {
        let _ = historical_var(&r, 0.05);
    }
}
