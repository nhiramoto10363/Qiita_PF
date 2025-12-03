#=
pf_julia.jl
粒子フィルタ Julia実装（マルチスレッド + SIMD最適化版）

実行:
  julia --threads=auto Julia/pf_julia.jl
=#

using Random
using Statistics
using SpecialFunctions: loggamma
using CSV
using DataFrames
using Base.Threads

# ============================================================
# 予測ステップ（並列化）
# ============================================================

function predict_linear!(particles::Vector{Float64}, noise::Vector{Float64})
    @inbounds @simd for i in eachindex(particles)
        particles[i] += noise[i]
    end
end

function predict_gordon!(particles::Vector{Float64}, noise::Vector{Float64}, t::Int)
    forcing = 8.0 * cos(1.2 * t)
    @inbounds @simd for i in eachindex(particles)
        x = particles[i]
        nonlinear = 25.0 * x / (1.0 + x * x)
        particles[i] = 0.5 * x + nonlinear + forcing + noise[i]
    end
end

function predict_positive_rw!(particles::Vector{Float64}, noise::Vector{Float64}, eps::Float64)
    @inbounds @simd for i in eachindex(particles)
        val = particles[i] + noise[i]
        particles[i] = val > eps ? val : eps
    end
end

# ============================================================
# 尤度計算（並列化）
# ============================================================

function loglik_gaussian!(loglik::Vector{Float64}, y::Float64, particles::Vector{Float64}, sigma::Float64)
    const_term = -0.5 * log(2π * sigma^2)
    inv_var = 1.0 / (sigma^2)
    @inbounds @simd for i in eachindex(particles)
        diff = y - particles[i]
        loglik[i] = const_term - 0.5 * diff^2 * inv_var
    end
end

function loglik_poisson!(loglik::Vector{Float64}, y::Float64, particles::Vector{Float64})
    log_gamma_y = loggamma(y + 1.0)
    @inbounds @simd for i in eachindex(particles)
        lam = max(particles[i], 1e-8)
        loglik[i] = y * log(lam) - lam - log_gamma_y
    end
end

# ============================================================
# 重み正規化
# ============================================================

function normalize_weights!(weights::Vector{Float64}, loglik::Vector{Float64})
    max_ll = maximum(loglik)
    
    @inbounds @simd for i in eachindex(weights)
        weights[i] = exp(loglik[i] - max_ll)
    end
    
    w_sum = sum(weights)
    if w_sum <= 0.0
        fill!(weights, 1.0 / length(weights))
    else
        @inbounds @simd for i in eachindex(weights)
            weights[i] /= w_sum
        end
    end
end

# ============================================================
# 重み付き平均
# ============================================================

function weighted_mean(particles::Vector{Float64}, weights::Vector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(particles)
        s += particles[i] * weights[i]
    end
    return s
end

# ============================================================
# システマティックリサンプリング
# ============================================================

function systematic_resample!(indices::Vector{Int}, weights::Vector{Float64}, u::Float64)
    n = length(weights)
    
    # 累積和
    cumsum_w = cumsum(weights)
    
    # リサンプリング
    j = 1
    for i in 1:n
        target = (i - 1 + u) / n
        while j < n && cumsum_w[j] < target
            j += 1
        end
        indices[i] = j
    end
end

function apply_resample!(particles::Vector{Float64}, particles_new::Vector{Float64}, indices::Vector{Int})
    @inbounds @simd for i in eachindex(particles)
        particles_new[i] = particles[indices[i]]
    end
    copyto!(particles, particles_new)
end

# ============================================================
# 粒子フィルタ メインループ
# ============================================================

function run_particle_filter(
    y::Vector{Float64},
    num_particles::Int,
    system::Symbol,  # :linear, :gordon, :positive_rw
    obs_type::Symbol,  # :gaussian, :poisson
    sigma_w::Float64,
    sigma_obs::Float64,
    seed::Int
)
    rng = MersenneTwister(seed)
    T = length(y)
    eps = 1e-8
    
    # 初期化
    particles = if system == :positive_rw
        abs.(3.0 .+ randn(rng, num_particles))
    else
        randn(rng, num_particles)
    end
    
    # 作業用配列
    noise = zeros(num_particles)
    loglik = zeros(num_particles)
    weights = zeros(num_particles)
    indices = zeros(Int, num_particles)
    particles_new = zeros(num_particles)
    
    x_hat = zeros(T)
    
    for t in 1:T
        # プロセスノイズ生成
        randn!(rng, noise)
        @inbounds @simd for i in 1:num_particles
            noise[i] *= sigma_w
        end
        
        # 予測ステップ
        if system == :linear
            predict_linear!(particles, noise)
        elseif system == :gordon
            predict_gordon!(particles, noise, t -1 )
        elseif system == :positive_rw
            predict_positive_rw!(particles, noise, eps)
        end
        
        # 尤度計算
        if obs_type == :gaussian
            loglik_gaussian!(loglik, y[t], particles, sigma_obs)
        elseif obs_type == :poisson
            loglik_poisson!(loglik, y[t], particles)
        end
        
        # 重み正規化
        normalize_weights!(weights, loglik)
        
        # 状態推定値
        x_hat[t] = weighted_mean(particles, weights)
        
        # リサンプリング
        u = rand(rng) / num_particles
        systematic_resample!(indices, weights, u)
        apply_resample!(particles, particles_new, indices)
    end
    
    return x_hat
end

# ============================================================
# RMSE計算
# ============================================================

function rmse(true_vals::Vector{Float64}, est_vals::Vector{Float64})
    sqrt(mean((true_vals .- est_vals).^2))
end

# ============================================================
# ベンチマーク実行
# ============================================================

function main()
    # データ読み込み
    data = CSV.read("Data/benchmark_data.csv", DataFrame)
    params = CSV.read("Data/benchmark_params.csv", DataFrame)
    
    y1 = Vector{Float64}(data.y1_obs)
    y2 = Vector{Float64}(data.y2_obs)
    y3 = Vector{Float64}(data.y3_obs)
    
    x1_true = Vector{Float64}(data.x1_true)
    x2_true = Vector{Float64}(data.x2_true)
    x3_true = Vector{Float64}(data.x3_true)
    
    num_particles = 10000
    n_runs = 5
    
    println("Julia Particle Filter Benchmark")
    println("================================")
    println("Number of threads: ", Threads.nthreads())
    println()
    
    results = DataFrame(
        case = String[],
        language = String[],
        num_particles = Int[],
        rmse = Float64[],
        time_mean_sec = Float64[],
        time_std_sec = Float64[]
    )
    
    # ウォームアップ（JITコンパイル）
    println("Warming up JIT compilation...")
    _ = run_particle_filter(y1[1:100], 100, :linear, :gaussian, 0.5, 1.0, 999)
    _ = run_particle_filter(y2[1:100], 100, :gordon, :gaussian, 0.3, 1.0, 999)
    _ = run_particle_filter(y3[1:100], 100, :positive_rw, :poisson, 0.2, 0.0, 999)
    println("Warmup complete.")
    println()
    
    # ケース1: 線形 + ガウシアン
    println("Case 1: Linear + Gaussian")
    sigma_w1 = params[1, :sigma_w]
    sigma_obs1 = params[1, :sigma_obs]
    times1 = Float64[]
    local xhat1
    for run in 1:n_runs
        t = @elapsed xhat1 = run_particle_filter(y1, num_particles, :linear, :gaussian, sigma_w1, sigma_obs1, run)
        push!(times1, t)
    end
    rmse1 = rmse(x1_true, xhat1)
    mean_time1 = mean(times1)
    std_time1 = std(times1, corrected=false)
    println("  RMSE: ", round(rmse1, digits=6))
    println("  Time: ", round(mean_time1, digits=4), " ± ", round(std_time1, digits=4), " sec")
    println()
    push!(results, ("case1_linear_gaussian", "Julia", num_particles, rmse1, mean_time1, std_time1))
    
    # ケース2: Gordon非線形 + ガウシアン
    println("Case 2: Gordon Nonlinear + Gaussian")
    sigma_w2 = params[2, :sigma_w]
    sigma_obs2 = params[2, :sigma_obs]
    times2 = Float64[]
    local xhat2
    for run in 1:n_runs
        t = @elapsed xhat2 = run_particle_filter(y2, num_particles, :gordon, :gaussian, sigma_w2, sigma_obs2, run)
        push!(times2, t)
    end
    rmse2 = rmse(x2_true, xhat2)
    mean_time2 = mean(times2)
    std_time2 = std(times2, corrected=false)
    println("  RMSE: ", round(rmse2, digits=6))
    println("  Time: ", round(mean_time2, digits=4), " ± ", round(std_time2, digits=4), " sec")
    println()
    push!(results, ("case2_gordon_nonlinear", "Julia", num_particles, rmse2, mean_time2, std_time2))
    
    # ケース3: 非負RW + Poisson
    println("Case 3: Positive RW + Poisson")
    sigma_w3 = params[3, :sigma_w]
    times3 = Float64[]
    local xhat3
    for run in 1:n_runs
        t = @elapsed xhat3 = run_particle_filter(y3, num_particles, :positive_rw, :poisson, sigma_w3, 0.0, run)
        push!(times3, t)
    end
    rmse3 = rmse(x3_true, xhat3)
    mean_time3 = mean(times3)
    std_time3 = std(times3, corrected=false)
    println("  RMSE: ", round(rmse3, digits=6))
    println("  Time: ", round(mean_time3, digits=4), " ± ", round(std_time3, digits=4), " sec")
    println()
    push!(results, ("case3_positive_rw_poisson", "Julia", num_particles, rmse3, mean_time3, std_time3))
    
    # 結果をCSV出力
    CSV.write("Results/julia.csv", results)
    
    println("==================================================")
    println(results)
    println("==================================================")
    println()
    println("Saved: results_julia.csv")
end

# 実行
main()