"""
粒子フィルタ言語比較用データ生成スクリプト

3ケースの時系列データを生成し、CSVファイルとして出力する。
Python / Rust / Fortran / Julia  の各実装で共通して使用。

ケース①: 線形システム + 線形観測 + 正規ノイズ
ケース②: Gordon型強非線形システム + 線形観測 + 正規ノイズ
ケース③: 非負ランダムウォーク + Poisson観測
"""

import numpy as np
import pandas as pd


def generate_case1_linear_gaussian(
    T: int = 10000,
    seed: int = 0,
    sigma_w: float = 0.5,
    sigma_obs: float = 1.0,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ケース①: 線形システム + 線形観測 + 正規ノイズ

    状態方程式: x_t = x_{t-1} + w_t,  w_t ~ N(0, σ_w²)
    観測方程式: y_t = x_t + v_t,      v_t ~ N(0, σ_obs²)
    """
    rng = np.random.default_rng(seed)

    x = np.empty(T)
    y = np.empty(T)

    x[0] = x0
    for t in range(1, T):
        x[t] = x[t - 1] + rng.normal(0.0, sigma_w)

    y[:] = x + rng.normal(0.0, sigma_obs, size=T)
    return x, y


def generate_case2_gordon_nonlinear(
    T: int = 10000,
    seed: int = 1,
    sigma_w: float = 0.3,
    sigma_obs: float = 1.0,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ケース②: Gordon型強非線形システム + 線形観測 + 正規ノイズ

    状態方程式:
        x_t = 0.5 * x_{t-1}
              + 25 * x_{t-1} / (1 + x_{t-1}²)
              + 8 * cos(1.2 * t)
              + w_t,  w_t ~ N(0, σ_w²)

    観測方程式: y_t = x_t + v_t,  v_t ~ N(0, σ_obs²)

    参考: Gordon, Salmond, Smith (1993)
    """
    rng = np.random.default_rng(seed)

    x = np.empty(T)
    y = np.empty(T)

    x[0] = x0
    for t in range(1, T):
        nonlinear = 25.0 * x[t - 1] / (1.0 + x[t - 1] ** 2)
        forcing = 8.0 * np.cos(1.2 * t)
        x[t] = 0.5 * x[t - 1] + nonlinear + forcing + rng.normal(0.0, sigma_w)

    y[:] = x + rng.normal(0.0, sigma_obs, size=T)
    return x, y


def generate_case3_poisson(
    T: int = 10000,
    seed: int = 2,
    sigma_w: float = 0.2,
    x0: float = 3.0,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ケース③: 非負ランダムウォーク + Poisson観測

    状態方程式: λ_t = max(λ_{t-1} + w_t, ε),  w_t ~ N(0, σ_w²)
    観測方程式: y_t ~ Poisson(λ_t)

    λ_t は観測のレート（強度）パラメータ。
    """
    rng = np.random.default_rng(seed)

    lam = np.empty(T)
    y = np.empty(T, dtype=np.int64)

    lam[0] = max(x0, eps)
    for t in range(1, T):
        lam_prop = lam[t - 1] + rng.normal(0.0, sigma_w)
        lam[t] = max(lam_prop, eps)

    y[:] = rng.poisson(lam=lam)
    return lam, y


def main():
    T = 10000

    # パラメータ設定
    params = {
        "case1": {"sigma_w": 0.5, "sigma_obs": 1.0},
        "case2": {"sigma_w": 0.3, "sigma_obs": 1.0},
        "case3": {"sigma_w": 0.2, "x0": 3.0},
    }

    # ケース①: 線形 + 正規
    x1, y1 = generate_case1_linear_gaussian(
        T=T, seed=0, sigma_w=params["case1"]["sigma_w"], sigma_obs=params["case1"]["sigma_obs"]
    )

    # ケース②: Gordon型非線形 + 正規
    x2, y2 = generate_case2_gordon_nonlinear(
        T=T, seed=1, sigma_w=params["case2"]["sigma_w"], sigma_obs=params["case2"]["sigma_obs"]
    )

    # ケース③: 非負RW + Poisson
    x3, y3 = generate_case3_poisson(T=T, seed=2, sigma_w=params["case3"]["sigma_w"], x0=params["case3"]["x0"])

    # CSV出力
    df = pd.DataFrame(
        {
            "t": np.arange(T),
            "x1_true": x1,
            "y1_obs": y1,
            "x2_true": x2,
            "y2_obs": y2,
            "x3_true": x3,
            "y3_obs": y3,
        }
    )
    df.to_csv("Data/benchmark_data.csv", index=False)

    # パラメータも保存（各言語実装で参照）
    params_df = pd.DataFrame(
        {
            "case": ["case1", "case2", "case3"],
            "sigma_w": [params["case1"]["sigma_w"], params["case2"]["sigma_w"], params["case3"]["sigma_w"]],
            "sigma_obs": [params["case1"]["sigma_obs"], params["case2"]["sigma_obs"], 0.0],  # case3はPoisson
            "system": ["linear", "gordon", "positive_rw"],
            "obs_type": ["gaussian", "gaussian", "poisson"],
        }
    )
    params_df.to_csv("Data/benchmark_params.csv", index=False)

    print(f"Generated {T} time steps for 3 cases")
    print(f"Saved: benchmark_data.csv, benchmark_params.csv")
    print(df.head(10))


if __name__ == "__main__":
    main()