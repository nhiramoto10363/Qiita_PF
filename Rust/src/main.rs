//! 粒子フィルタ Rust実装（最適化版：Single Thread + Buffer Reuse）
//!
//! コンパイル設定 (Cargo.toml):
//! [dependencies]
//! rand = "0.8"
//! rand_distr = "0.4"
//! csv = "1.3"
//! serde = { version = "1.0", features = ["derive"] }
//!
//! 実行コマンド:
//! RUSTFLAGS="-C target-cpu=native" cargo run --release

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// ベンチマークデータの1行
#[derive(Debug, Deserialize)]
struct DataRow {
    t: i64,
    x1_true: f64,
    y1_obs: f64,
    x2_true: f64,
    y2_obs: f64,
    x3_true: f64,
    y3_obs: f64,
}

/// パラメータデータの1行
#[derive(Debug, Deserialize)]
struct ParamRow {
    case: String,
    sigma_w: f64,
    sigma_obs: f64,
    system: String,
    obs_type: String,
}

#[derive(Clone, Copy)]
enum SystemModel {
    Linear,
    Gordon,
    PositiveRW,
}

#[derive(Clone, Copy)]
enum ObsModel {
    Gaussian,
    Poisson,
}

// -----------------------------------------------------------------------------
// 予測ステップ (In-place更新、Rayonなし)
// -----------------------------------------------------------------------------

fn predict_linear(particles: &mut [f64], noise: &[f64]) {
    // 単純なループはLLVMが強力にベクトル化します
    for (p, &n) in particles.iter_mut().zip(noise.iter()) {
        *p += n;
    }
}

fn predict_gordon(particles: &mut [f64], noise: &[f64], t: usize) {
    let forcing = 8.0 * (1.2 * t as f64).cos();
    for (p, &n) in particles.iter_mut().zip(noise.iter()) {
        let x = *p;
        let nonlinear = 25.0 * x / (1.0 + x * x);
        *p = 0.5 * x + nonlinear + forcing + n;
    }
}

fn predict_positive_rw(particles: &mut [f64], noise: &[f64], eps: f64) {
    for (p, &n) in particles.iter_mut().zip(noise.iter()) {
        let val = *p + n;
        *p = if val > eps { val } else { eps };
    }
}

// -----------------------------------------------------------------------------
// 尤度計算 (Buffer Reuse)
// 結果を返すのではなく、渡された loglik バッファに書き込む
// -----------------------------------------------------------------------------

fn loglik_gaussian(y: f64, particles: &[f64], sigma: f64, loglik: &mut [f64]) {
    // 定数計算をループ外へ
    let const_term = -0.5 * (2.0 * PI * sigma * sigma).ln();
    let inv_var = 1.0 / (sigma * sigma);

    for (i, &p) in particles.iter().enumerate() {
        let diff = y - p;
        loglik[i] = const_term - 0.5 * diff * diff * inv_var;
    }
}

fn loglik_poisson(y: f64, particles: &[f64], loglik: &mut [f64]) {
    let log_gamma_y = ln_gamma(y + 1.0);

    for (i, &p) in particles.iter().enumerate() {
        let lam = if p < 1e-8 { 1e-8 } else { p };
        loglik[i] = y * lam.ln() - lam - log_gamma_y;
    }
}

// -----------------------------------------------------------------------------
// 数学関数
// -----------------------------------------------------------------------------

#[inline(always)]
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let z = 1.0 - x;
        PI.ln() - (PI * x).sin().ln() - ln_gamma(z)
    } else {
        let z = x - 1.0;
        let mut sum = C[0];
        for i in 1..9 {
            sum += C[i] / (z + i as f64);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * PI).ln() + (t.ln() * (z + 0.5)) - t + sum.ln()
    }
}

// -----------------------------------------------------------------------------
// 重み計算 (Buffer Reuse)
// -----------------------------------------------------------------------------

fn normalize_weights(loglik: &[f64], weights: &mut [f64]) {
    // max_llの探索 (イテレータ使用)
    let max_ll = loglik
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| if v > m { v } else { m });

    let mut sum = 0.0;
    for (i, &ll) in loglik.iter().enumerate() {
        let w = (ll - max_ll).exp();
        weights[i] = w;
        sum += w;
    }

    // 正規化
    if sum <= 0.0 {
        let inv_n = 1.0 / weights.len() as f64;
        for w in weights.iter_mut() {
            *w = inv_n;
        }
    } else {
        let inv_sum = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv_sum;
        }
    }
}

// -----------------------------------------------------------------------------
// リサンプリング
// -----------------------------------------------------------------------------

fn weighted_mean(particles: &[f64], weights: &[f64]) -> f64 {
    // イテレータのzipは高速
    particles
        .iter()
        .zip(weights.iter())
        .map(|(&p, &w)| p * w)
        .sum()
}

fn systematic_resample(weights: &[f64], u: f64, indices: &mut [usize]) {
    let n = weights.len();
    
    // 累積和を計算しながらリサンプリング位置を探す（メモリ節約のためcumsum配列を作らない）
    let mut cum_weight = weights[0];
    let mut i = 0;
    
    for j in 0..n {
        let target = (j as f64 + u) / n as f64;
        while i < n - 1 && cum_weight < target {
            i += 1;
            cum_weight += weights[i];
        }
        indices[j] = i;
    }
}

fn apply_resample(particles: &mut [f64], particles_new: &mut [f64], indices: &[usize]) {
    // 新しいバッファにコピー
    for (j, &idx) in indices.iter().enumerate() {
        particles_new[j] = particles[idx];
    }
    // 元のバッファに書き戻す (copy_from_slice)
    particles.copy_from_slice(particles_new);
}

// -----------------------------------------------------------------------------
// メイン処理
// -----------------------------------------------------------------------------

fn run_particle_filter(
    y: &[f64],
    num_particles: usize,
    system: SystemModel,
    obs_model: ObsModel,
    sigma_w: f64,
    sigma_obs: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal_proc = Normal::new(0.0, sigma_w).unwrap();
    let normal_init = Normal::new(0.0, 1.0).unwrap();
    
    let t_len = y.len();
    let eps = 1e-8;

    // --- メモリ確保 (ループ外で一度だけ行う) ---
    // これによりT回のループ中でのアロケーションをゼロにする
    let mut particles: Vec<f64> = vec![0.0; num_particles];
    let mut noise: Vec<f64> = vec![0.0; num_particles];
    let mut loglik: Vec<f64> = vec![0.0; num_particles];
    let mut weights: Vec<f64> = vec![0.0; num_particles];
    let mut indices: Vec<usize> = vec![0; num_particles];
    let mut particles_new: Vec<f64> = vec![0.0; num_particles];
    let mut x_hat = vec![0.0; t_len];

    // 初期粒子生成
    match system {
        SystemModel::PositiveRW => {
            for p in particles.iter_mut() {
                // 修正箇所: ここで型を明示しないとabs()が呼べない
                let x: f64 = 3.0 + normal_init.sample(&mut rng);
                *p = x.abs();
            }
        },
        _ => {
            for p in particles.iter_mut() {
                *p = normal_init.sample(&mut rng);
            }
        }
    }

    // --- メインループ ---
    for t in 0..t_len {
        // ノイズ生成 (Buffer Reuse)
        for n in noise.iter_mut() {
            *n = normal_proc.sample(&mut rng);
        }

        // 予測
        match system {
            SystemModel::Linear => predict_linear(&mut particles, &noise),
            SystemModel::Gordon => predict_gordon(&mut particles, &noise, t),
            SystemModel::PositiveRW => predict_positive_rw(&mut particles, &noise, eps),
        }

        // 尤度
        match obs_model {
            ObsModel::Gaussian => loglik_gaussian(y[t], &particles, sigma_obs, &mut loglik),
            ObsModel::Poisson => loglik_poisson(y[t], &particles, &mut loglik),
        }

        // 重み
        normalize_weights(&loglik, &mut weights);

        // 推定
        x_hat[t] = weighted_mean(&particles, &weights);

        // リサンプリング
        let u: f64 = rng.gen::<f64>() / num_particles as f64; // uniform [0, 1/N]
        systematic_resample(&weights, u, &mut indices);
        apply_resample(&mut particles, &mut particles_new, &indices);
    }

    x_hat
}

// -----------------------------------------------------------------------------
// ユーティリティ
// -----------------------------------------------------------------------------

fn rmse(true_vals: &[f64], est_vals: &[f64]) -> f64 {
    let mse: f64 = true_vals
        .iter()
        .zip(est_vals.iter())
        .map(|(&t, &e)| (t - e).powi(2))
        .sum::<f64>()
        / true_vals.len() as f64;
    mse.sqrt()
}

fn load_data(path: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(path).expect("Cannot open data file");
    let mut reader = csv::Reader::from_reader(file);

    let mut x1_true = Vec::new();
    let mut y1_obs = Vec::new();
    let mut x2_true = Vec::new();
    let mut y2_obs = Vec::new();
    let mut x3_true = Vec::new();
    let mut y3_obs = Vec::new();

    for result in reader.deserialize() {
        let row: DataRow = result.expect("Cannot parse row");
        x1_true.push(row.x1_true);
        y1_obs.push(row.y1_obs);
        x2_true.push(row.x2_true);
        y2_obs.push(row.y2_obs);
        x3_true.push(row.x3_true);
        y3_obs.push(row.y3_obs);
    }

    (x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs)
}

fn load_params(path: &str) -> Vec<ParamRow> {
    let file = File::open(path).expect("Cannot open params file");
    let mut reader = csv::Reader::from_reader(file);
    reader.deserialize().map(|r| r.expect("Cannot parse param row")).collect()
}

fn main() {
    let (x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs) = load_data("../Data/benchmark_data.csv");
    let params = load_params("../Data/benchmark_params.csv");

    let num_particles = 10000;
    let n_runs = 5;

    println!("Rust Particle Filter Benchmark (Optimized: Single Thread + No Alloc)");
    println!("===================================================================");

    let mut results = Vec::new();

    // Case 1
    println!("Case 1: Linear + Gaussian");
    let p1 = &params[0];
    let mut times1 = Vec::new();
    let mut xhat1 = Vec::new();
    for run in 0..n_runs {
        let start = Instant::now();
        xhat1 = run_particle_filter(&y1_obs, num_particles, SystemModel::Linear, ObsModel::Gaussian, p1.sigma_w, p1.sigma_obs, run as u64);
        times1.push(start.elapsed().as_secs_f64());
    }
    let rmse1 = rmse(&x1_true, &xhat1);
    let mean1 = times1.iter().sum::<f64>() / n_runs as f64;
    let std1 = (times1.iter().map(|t| (t - mean1).powi(2)).sum::<f64>() / n_runs as f64).sqrt();
    println!("  RMSE: {:.6}", rmse1);
    println!("  Time: {:.4} ± {:.4} sec\n", mean1, std1);
    results.push(("case1_linear_gaussian", rmse1, mean1, std1));

    // Case 2
    println!("Case 2: Gordon Nonlinear + Gaussian");
    let p2 = &params[1];
    let mut times2 = Vec::new();
    let mut xhat2 = Vec::new();
    for run in 0..n_runs {
        let start = Instant::now();
        xhat2 = run_particle_filter(&y2_obs, num_particles, SystemModel::Gordon, ObsModel::Gaussian, p2.sigma_w, p2.sigma_obs, run as u64);
        times2.push(start.elapsed().as_secs_f64());
    }
    let rmse2 = rmse(&x2_true, &xhat2);
    let mean2 = times2.iter().sum::<f64>() / n_runs as f64;
    let std2 = (times2.iter().map(|t| (t - mean2).powi(2)).sum::<f64>() / n_runs as f64).sqrt();
    println!("  RMSE: {:.6}", rmse2);
    println!("  Time: {:.4} ± {:.4} sec\n", mean2, std2);
    results.push(("case2_gordon_nonlinear", rmse2, mean2, std2));

    // Case 3
    println!("Case 3: Positive RW + Poisson");
    let p3 = &params[2];
    let mut times3 = Vec::new();
    let mut xhat3 = Vec::new();
    for run in 0..n_runs {
        let start = Instant::now();
        xhat3 = run_particle_filter(&y3_obs, num_particles, SystemModel::PositiveRW, ObsModel::Poisson, p3.sigma_w, 0.0, run as u64);
        times3.push(start.elapsed().as_secs_f64());
    }
    let rmse3 = rmse(&x3_true, &xhat3);
    let mean3 = times3.iter().sum::<f64>() / n_runs as f64;
    let std3 = (times3.iter().map(|t| (t - mean3).powi(2)).sum::<f64>() / n_runs as f64).sqrt();
    println!("  RMSE: {:.6}", rmse3);
    println!("  Time: {:.4} ± {:.4} sec\n", mean3, std3);
    results.push(("case3_positive_rw_poisson", rmse3, mean3, std3));

    // Save
    let mut file = File::create("../Results/rust.csv").expect("Cannot create results file");
    writeln!(file, "case,language,num_particles,rmse,time_mean_sec,time_std_sec").unwrap();
    for (case, r, m, s) in &results {
        writeln!(file, "{},Rust (Optimized),{},{:.6},{:.6},{:.6}", case, num_particles, r, m, s).unwrap();
    }
    println!("Saved: results_rust.csv");
}