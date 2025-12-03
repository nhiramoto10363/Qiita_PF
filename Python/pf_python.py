"""
粒子フィルタ Python実装（Numba SIMD最適化版）

【診断結果による最適化】
並列化(parallel=True)は N=10,000 程度ではオーバーヘッドが勝るため採用しません。
代わりに fastmath=True による強力なベクトル化(SIMD)で高速化します。
"""

import time
import numpy as np
import pandas as pd
from numba import njit
from math import lgamma, pi, log, exp, cos

# コンパイルオプション
# parallel=False: スレッドオーバーヘッド回避
# fastmath=True: Juliaの@simd相当の最適化
# cache=True: 次回以降の起動高速化
JIT_OPTS = {'fastmath': True, 'parallel': False, 'cache': True}

# ---- 予測ステップ（in-place, vectorized）----

@njit(**JIT_OPTS)
def predict_linear_inplace(particles, noise):
    # 単純加算はNumbaが自動でAVX命令等に変換する
    particles += noise


@njit(**JIT_OPTS)
def predict_gordon_inplace(particles, noise, t):
    forcing = 8.0 * cos(1.2 * t)
    # テンポラリ配列を作らせないよう、計算式をまとめる
    # Numbaは式全体を見てループ融合してくれる
    particles[:] = (
        0.5 * particles
        + 25.0 * particles / (1.0 + particles * particles)
        + forcing
        + noise
    )


@njit(**JIT_OPTS)
def predict_positive_rw_inplace(particles, noise, eps):
    particles += noise
    # np.maximum は ufunc として高速に動作
    particles[:] = np.maximum(particles, eps)


# ---- 尤度計算（in-place, vectorized）----

@njit(**JIT_OPTS)
def loglik_gaussian_inplace(y_t, particles, sigma, out_loglik):
    const = -0.5 * log(2.0 * pi * sigma * sigma)
    inv_var = 1.0 / (sigma * sigma)

    # メモリアロケーションを避けるため out_loglik を計算バッファとして利用
    out_loglik[:] = y_t - particles
    out_loglik[:] = const - 0.5 * out_loglik * out_loglik * inv_var


@njit(**JIT_OPTS)
def loglik_poisson_inplace(y_t, particles, out_loglik):
    log_gamma_y = lgamma(y_t + 1.0)

    out_loglik[:] = particles
    out_loglik[:] = np.maximum(out_loglik, 1e-8)
    out_loglik[:] = y_t * np.log(out_loglik) - out_loglik - log_gamma_y


# ---- 重み正規化（in-place）----

@njit(**JIT_OPTS)
def normalize_weights_inplace(loglik, weights):
    max_ll = np.max(loglik)
    weights[:] = np.exp(loglik - max_ll)
    w_sum = np.sum(weights)
    
    if w_sum <= 0.0:
        n = weights.shape[0]
        weights[:] = 1.0 / n
    else:
        weights /= w_sum


@njit(**JIT_OPTS)
def weighted_mean(particles, weights):
    return float(np.dot(particles, weights))


# ---- リサンプリング（シーケンシャルが最速）----

@njit(**JIT_OPTS)
def systematic_resample_inplace(weights, u, indices):
    """
    累積和配列(cumsum)を確保・計算するコストすら削る実装。
    累積和をオンザフライで計算しながら走査する。
    """
    n = weights.shape[0]
    cum_weight = weights[0]
    i = 0
    
    for j in range(n):
        target = (j + u) / n
        while i < n - 1 and cum_weight < target:
            i += 1
            cum_weight += weights[i]
        indices[j] = i


@njit(**JIT_OPTS)
def apply_resample_inplace(particles, indices, tmp_particles):
    # ファンシーインデックス参照はNumbaが得意とするところ
    tmp_particles[:] = particles[indices]
    particles[:] = tmp_particles


# ============================================================
# メインループ
# ============================================================

@njit(**JIT_OPTS)
def run_particle_filter_jit(
    y,
    num_particles,
    system_id, # 0=linear, 1=gordon, 2=positive_rw
    obs_id,    # 0=gaussian, 1=poisson
    sigma_w,
    sigma_obs,
    seed,
):
    T = len(y)
    x_hat = np.empty(T)
    eps = 1e-8

    np.random.seed(seed)

    # 初期粒子生成
    if system_id == 2:
        particles = np.abs(3.0 + np.random.normal(0.0, 1.0, num_particles))
    else:
        particles = np.random.normal(0.0, 1.0, num_particles)

    # アロケーションはループ外で1回のみ（Julia/Rustと同等の戦略）
    noise = np.empty(num_particles)
    loglik = np.empty(num_particles)
    weights = np.empty(num_particles)
    indices = np.empty(num_particles, dtype=np.int64)
    tmp_particles = np.empty(num_particles)

    for t in range(T):
        # 1. ノイズ生成 (in-place)
        # Numbaのnp.random.normalは高速だが、配列作成を避けるため
        # out引数を使いたいが、numpy標準APIと違いNumba版はout引数をサポートしていない場合がある。
        # しかし noise[:] = ... で代入すればメモリ再利用されるよう最適化される。
        noise[:] = np.random.normal(0.0, sigma_w, num_particles)

        # 2. 予測
        if system_id == 0:
            predict_linear_inplace(particles, noise)
        elif system_id == 1:
            predict_gordon_inplace(particles, noise, t)
        else:
            predict_positive_rw_inplace(particles, noise, eps)

        # 3. 尤度
        if obs_id == 0:
            loglik_gaussian_inplace(y[t], particles, sigma_obs, loglik)
        else:
            loglik_poisson_inplace(y[t], particles, loglik)

        # 4. 重み & 推定
        normalize_weights_inplace(loglik, weights)
        x_hat[t] = weighted_mean(particles, weights)

        # 5. リサンプリング
        u = np.random.uniform(0.0, 1.0 / num_particles)
        systematic_resample_inplace(weights, u, indices)
        apply_resample_inplace(particles, indices, tmp_particles)

    return x_hat


def run_particle_filter(
    y: np.ndarray,
    num_particles: int,
    system: str,
    obs_type: str,
    sigma_w: float,
    sigma_obs: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    
    sys_map = {"linear": 0, "gordon": 1, "positive_rw": 2}
    obs_map = {"gaussian": 0, "poisson": 1}

    return run_particle_filter_jit(
        y.astype(np.float64),
        num_particles,
        sys_map[system],
        obs_map[obs_type],
        float(sigma_w),
        float(sigma_obs),
        int(seed),
    )


# ============================================================
# ベンチマーク実行用
# ============================================================

def rmse(true: np.ndarray, est: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true - est) ** 2)))

def main():
    # データの読み込み
    try:
        df = pd.read_csv("Data/benchmark_data.csv")
        params_df = pd.read_csv("Data/benchmark_params.csv")
    except FileNotFoundError:
        print("Error: benchmark_data.csv or benchmark_params.csv not found.")
        return

    y1 = df["y1_obs"].values
    y2 = df["y2_obs"].values
    y3 = df["y3_obs"].values.astype(float)
    x1_true = df["x1_true"].values
    x2_true = df["x2_true"].values
    x3_true = df["x3_true"].values

    p1 = params_df[params_df["case"] == "case1"].iloc[0]
    p2 = params_df[params_df["case"] == "case2"].iloc[0]
    p3 = params_df[params_df["case"] == "case3"].iloc[0]

    num_particles = 10000
    n_runs = 5

    print(f"Python (Numba) Particle Filter Benchmark")
    print(f"========================================")
    print(f"Mode: Single Thread + SIMD (parallel=False)")
    
    # Warmup
    print("Warming up JIT...")
    run_particle_filter(np.random.randn(100), 100, "linear", "gaussian", 0.5)
    run_particle_filter(np.random.randn(100), 100, "gordon", "gaussian", 0.5)
    run_particle_filter(np.abs(np.random.randn(100)), 100, "positive_rw", "poisson", 0.5)
    print("Warmup complete.\n")

    results = []

    # Case 1
    print("Case 1: Linear + Gaussian")
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        xh = run_particle_filter(y1, num_particles, "linear", "gaussian", p1["sigma_w"], p1["sigma_obs"], r)
        times.append(time.perf_counter() - t0)
    
    r_val = rmse(x1_true, xh)
    t_mean = np.mean(times)
    t_std = np.std(times)
    print(f"  RMSE: {r_val:.6f}")
    print(f"  Time: {t_mean:.4f} ± {t_std:.4f} sec\n")
    results.append(["case1_linear_gaussian", "Python(Numba)", num_particles, r_val, t_mean, t_std])

    # Case 2
    print("Case 2: Gordon Nonlinear + Gaussian")
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        xh = run_particle_filter(y2, num_particles, "gordon", "gaussian", p2["sigma_w"], p2["sigma_obs"], r)
        times.append(time.perf_counter() - t0)
    
    r_val = rmse(x2_true, xh)
    t_mean = np.mean(times)
    t_std = np.std(times)
    print(f"  RMSE: {r_val:.6f}")
    print(f"  Time: {t_mean:.4f} ± {t_std:.4f} sec\n")
    results.append(["case2_gordon_nonlinear", "Python(Numba)", num_particles, r_val, t_mean, t_std])

    # Case 3
    print("Case 3: Positive RW + Poisson")
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        xh = run_particle_filter(y3, num_particles, "positive_rw", "poisson", p3["sigma_w"], 0.0, r)
        times.append(time.perf_counter() - t0)
    
    r_val = rmse(x3_true, xh)
    t_mean = np.mean(times)
    t_std = np.std(times)
    print(f"  RMSE: {r_val:.6f}")
    print(f"  Time: {t_mean:.4f} ± {t_std:.4f} sec\n")
    results.append(["case3_positive_rw_poisson", "Python(Numba)", num_particles, r_val, t_mean, t_std])

    # Save
    res_df = pd.DataFrame(results, columns=["case", "language", "num_particles", "rmse", "time_mean_sec", "time_std_sec"])
    res_df.to_csv("Results/python.csv", index=False)

if __name__ == "__main__":
    main()