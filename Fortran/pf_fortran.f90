! pf_fortran.f90
! 粒子フィルタ Fortran実装（OpenMP並列化版）
!
! コンパイル:
!   export OMP_NUM_THREADS=1
!   gfortran -O3 -march=native -ffast-math -fopenmp -march=native Fortran/pf_fortran.f90 -o Fortran/pf_fortran
!
! 実行:
!   ./pf_fortran

program particle_filter_benchmark
    use omp_lib
    implicit none

    integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: T = 10000
    integer, parameter :: num_particles = 10000
    integer, parameter :: n_runs = 5

    real(dp), allocatable :: x1_true(:), y1_obs(:)
    real(dp), allocatable :: x2_true(:), y2_obs(:)
    real(dp), allocatable :: x3_true(:), y3_obs(:)
    real(dp), allocatable :: xhat(:)

    real(dp) :: sigma_w(3), sigma_obs(3)
    real(dp) :: times(n_runs), rmse_val
    real(dp) :: mean_time, std_time
    real(dp) :: t_start, t_end
    integer :: run, i, io_unit

    ! パラメータ設定
    sigma_w = [0.5d0, 0.3d0, 0.2d0]
    sigma_obs = [1.0d0, 1.0d0, 0.0d0]

    ! メモリ確保
    allocate(x1_true(T), y1_obs(T))
    allocate(x2_true(T), y2_obs(T))
    allocate(x3_true(T), y3_obs(T))
    allocate(xhat(T))

    ! データ読み込み
    call load_data('Data/benchmark_data.csv', x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs, T)

    print '(A)', 'Fortran Particle Filter Benchmark (OpenMP)'
    print '(A)', '=========================================='
    print '(A,I0)', 'Number of threads: ', omp_get_max_threads()
    print '(A)', ''

    ! 結果ファイルを開く
    open(newunit=io_unit, file='Results/fortran.csv', status='replace')
    write(io_unit, '(A)') 'case,language,num_particles,rmse,time_mean_sec,time_std_sec'

    ! ケース1: 線形 + ガウシアン
    print '(A)', 'Case 1: Linear + Gaussian'
    do run = 1, n_runs
        t_start = omp_get_wtime()
        call run_pf(y1_obs, T, num_particles, 1, 1, sigma_w(1), sigma_obs(1), run, xhat)
        t_end = omp_get_wtime()
        times(run) = t_end - t_start
    end do
    rmse_val = calc_rmse(x1_true, xhat, T)
    call calc_stats(times, n_runs, mean_time, std_time)
    print '(A,F10.6)', '  RMSE: ', rmse_val
    print '(A,F8.4,A,F8.4,A)', '  Time: ', mean_time, ' ± ', std_time, ' sec'
    print '(A)', ''
    write(io_unit, '(A,I0,A,F10.6,A,F10.6,A,F10.6)') &
        'case1_linear_gaussian,Fortran (OpenMP),', num_particles, ',', rmse_val, ',', mean_time, ',', std_time

    ! ケース2: Gordon非線形 + ガウシアン
    print '(A)', 'Case 2: Gordon Nonlinear + Gaussian'
    do run = 1, n_runs
        t_start = omp_get_wtime()
        call run_pf(y2_obs, T, num_particles, 2, 1, sigma_w(2), sigma_obs(2), run, xhat)
        t_end = omp_get_wtime()
        times(run) = t_end - t_start
    end do
    rmse_val = calc_rmse(x2_true, xhat, T)
    call calc_stats(times, n_runs, mean_time, std_time)
    print '(A,F10.6)', '  RMSE: ', rmse_val
    print '(A,F8.4,A,F8.4,A)', '  Time: ', mean_time, ' ± ', std_time, ' sec'
    print '(A)', ''
    write(io_unit, '(A,I0,A,F10.6,A,F10.6,A,F10.6)') &
        'case2_gordon_nonlinear,Fortran (OpenMP),', num_particles, ',', rmse_val, ',', mean_time, ',', std_time

    ! ケース3: 非負RW + Poisson
    print '(A)', 'Case 3: Positive RW + Poisson'
    do run = 1, n_runs
        t_start = omp_get_wtime()
        call run_pf(y3_obs, T, num_particles, 3, 2, sigma_w(3), 0.0d0, run, xhat)
        t_end = omp_get_wtime()
        times(run) = t_end - t_start
    end do
    rmse_val = calc_rmse(x3_true, xhat, T)
    call calc_stats(times, n_runs, mean_time, std_time)
    print '(A,F10.6)', '  RMSE: ', rmse_val
    print '(A,F8.4,A,F8.4,A)', '  Time: ', mean_time, ' ± ', std_time, ' sec'
    print '(A)', ''
    write(io_unit, '(A,I0,A,F10.6,A,F10.6,A,F10.6)') &
        'case3_positive_rw_poisson,Fortran (OpenMP),', num_particles, ',', rmse_val, ',', mean_time, ',', std_time

    close(io_unit)
    print '(A)', 'Saved: results_fortran.csv'

    deallocate(x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs, xhat)

contains

    !---------------------------------------------------------------------------
    ! 粒子フィルタメインルーチン
    !---------------------------------------------------------------------------
    subroutine run_pf(y, T, np, system, obs_type, sigma_w, sigma_obs, seed, xhat)
        integer, intent(in) :: T, np, system, obs_type, seed
        real(dp), intent(in) :: y(T), sigma_w, sigma_obs
        real(dp), intent(out) :: xhat(T)

        real(dp), allocatable :: particles(:), noise(:), loglik(:), weights(:)
        real(dp), allocatable :: particles_new(:)
        integer, allocatable :: indices(:)
        real(dp), allocatable :: cumsum(:)
        real(dp) :: u, eps
        integer :: t_idx, i

        allocate(particles(np), noise(np), loglik(np), weights(np))
        allocate(particles_new(np), indices(np))
        allocate(cumsum(np))

        eps = 1.0d-8

        ! 乱数シード初期化
        call init_random(seed)

        ! 初期粒子
        if (system == 3) then
            call randn_vec(particles, np)
            !$omp parallel do
            do i = 1, np
                particles(i) = abs(3.0d0 + particles(i))
            end do
            !$omp end parallel do
        else
            call randn_vec(particles, np)
        end if

        do t_idx = 1, T
            ! プロセスノイズ生成
            call randn_vec(noise, np)
            !$omp parallel do
            do i = 1, np
                noise(i) = noise(i) * sigma_w
            end do
            !$omp end parallel do

            ! 予測ステップ
            select case (system)
            case (1)  ! linear
                call predict_linear(particles, noise, np)
            case (2)  ! gordon
                call predict_gordon(particles, noise, np, t_idx-1)
            case (3)  ! positive_rw
                call predict_positive_rw(particles, noise, np, eps)
            end select

            ! 尤度計算
            select case (obs_type)
            case (1)  ! gaussian
                call loglik_gaussian(y(t_idx), particles, sigma_obs, np, loglik)
            case (2)  ! poisson
                call loglik_poisson(y(t_idx), particles, np, loglik)
            end select

            ! 重み正規化
            call normalize_weights(loglik, np, weights)

            ! 状態推定値
            xhat(t_idx) = weighted_mean(particles, weights, np)

            ! リサンプリング
            call random_number(u)
            u = u / np
            call systematic_resample(weights, np, u, indices, cumsum)
            call apply_resample(particles, indices, np, particles_new)
            call swap_alloc(particles, particles_new)

        end do

        deallocate(particles, noise, loglik, weights, particles_new, indices, cumsum)
    end subroutine run_pf

    !---------------------------------------------------------------------------
    ! 予測ステップ
    !---------------------------------------------------------------------------
    subroutine predict_linear(particles, noise, np)
        integer, intent(in) :: np
        real(dp), intent(inout) :: particles(np)
        real(dp), intent(in) :: noise(np)
        integer :: i

        !$omp parallel do
        do i = 1, np
            particles(i) = particles(i) + noise(i)
        end do
        !$omp end parallel do
    end subroutine predict_linear

    subroutine predict_gordon(particles, noise, np, t)
        integer, intent(in) :: np, t
        real(dp), intent(inout) :: particles(np)
        real(dp), intent(in) :: noise(np)
        real(dp) :: forcing, x, nonlinear
        integer :: i

        forcing = 8.0d0 * cos(1.2d0 * t)

        !$omp parallel do private(x, nonlinear)
        do i = 1, np
            x = particles(i)
            nonlinear = 25.0d0 * x / (1.0d0 + x*x)
            particles(i) = 0.5d0 * x + nonlinear + forcing + noise(i)
        end do
        !$omp end parallel do
    end subroutine predict_gordon

    subroutine predict_positive_rw(particles, noise, np, eps)
        integer, intent(in) :: np
        real(dp), intent(inout) :: particles(np)
        real(dp), intent(in) :: noise(np), eps
        real(dp) :: val
        integer :: i

        !$omp parallel do private(val)
        do i = 1, np
            val = particles(i) + noise(i)
            if (val > eps) then
                particles(i) = val
            else
                particles(i) = eps
            end if
        end do
        !$omp end parallel do
    end subroutine predict_positive_rw

    !---------------------------------------------------------------------------
    ! 尤度計算
    !---------------------------------------------------------------------------
    subroutine loglik_gaussian(y, particles, sigma, np, loglik)
        integer, intent(in) :: np
        real(dp), intent(in) :: y, particles(np), sigma
        real(dp), intent(out) :: loglik(np)
        real(dp), parameter :: pi = 3.141592653589793d0
        real(dp) :: const_term, inv_var, diff
        integer :: i

        const_term = -0.5d0 * log(2.0d0 * pi * sigma * sigma)
        inv_var = 1.0d0 / (sigma * sigma)

        !$omp parallel do private(diff)
        do i = 1, np
            diff = y - particles(i)
            loglik(i) = const_term - 0.5d0 * diff * diff * inv_var
        end do
        !$omp end parallel do
    end subroutine loglik_gaussian

    subroutine loglik_poisson(y, particles, np, loglik)
        integer, intent(in) :: np
        real(dp), intent(in) :: y, particles(np)
        real(dp), intent(out) :: loglik(np)
        real(dp) :: log_gamma_y, lam
        integer :: i

        log_gamma_y = log_gamma(y + 1.0d0)

        !$omp parallel do private(lam)
        do i = 1, np
            lam = particles(i)
            if (lam < 1.0d-8) lam = 1.0d-8
            loglik(i) = y * log(lam) - lam - log_gamma_y
        end do
        !$omp end parallel do
    end subroutine loglik_poisson

    !---------------------------------------------------------------------------
    ! 重み正規化
    !---------------------------------------------------------------------------
    subroutine normalize_weights(loglik, np, weights)
        integer, intent(in) :: np
        real(dp), intent(in) :: loglik(np)
        real(dp), intent(out) :: weights(np)
        real(dp) :: max_ll, w_sum
        integer :: i

        max_ll = maxval(loglik)

        w_sum = 0.0d0
        !$omp parallel do reduction(+:w_sum)
        do i = 1, np
            weights(i) = exp(loglik(i) - max_ll)
            w_sum = w_sum + weights(i)
        end do
        !$omp end parallel do

        if (w_sum <= 0.0d0) then
            weights = 1.0d0 / np
        else
            !$omp parallel do
            do i = 1, np
                weights(i) = weights(i) / w_sum
            end do
            !$omp end parallel do
        end if
    end subroutine normalize_weights

    !---------------------------------------------------------------------------
    ! 重み付き平均
    !---------------------------------------------------------------------------
    function weighted_mean(particles, weights, np) result(mean)
        integer, intent(in) :: np
        real(dp), intent(in) :: particles(np), weights(np)
        real(dp) :: mean
        integer :: i

        mean = 0.0d0
        !$omp parallel do reduction(+:mean)
        do i = 1, np
            mean = mean + particles(i) * weights(i)
        end do
        !$omp end parallel do
    end function weighted_mean

    !---------------------------------------------------------------------------
    ! システマティックリサンプリング
    !---------------------------------------------------------------------------
    subroutine systematic_resample(weights, np, u, indices, cumsum)
        integer, intent(in) :: np
        real(dp), intent(in) :: weights(np), u
        integer, intent(out) :: indices(np)
        real(dp), intent(out) :: cumsum(np)
        real(dp) :: target
        integer :: i, j
    
        cumsum(1) = weights(1)
        do i = 2, np
            cumsum(i) = cumsum(i-1) + weights(i)
        end do
    
        i = 1
        do j = 1, np
            target = (j - 1 + u) / np
            do while (i < np .and. cumsum(i) < target)
                i = i + 1
            end do
            indices(j) = i
        end do
    end subroutine systematic_resample


    subroutine apply_resample(particles, indices, np, particles_new)
        integer, intent(in) :: np
        real(dp), intent(in) :: particles(np)
        integer, intent(in) :: indices(np)
        real(dp), intent(out) :: particles_new(np)
        integer :: i

        !$omp parallel do
        do i = 1, np
            particles_new(i) = particles(indices(i))
        end do
        !$omp end parallel do
    end subroutine apply_resample

    !---------------------------------------------------------------------------
    ! ユーティリティ
    !---------------------------------------------------------------------------
    function calc_rmse(true_vals, est_vals, n) result(rmse)
        integer, intent(in) :: n
        real(dp), intent(in) :: true_vals(n), est_vals(n)
        real(dp) :: rmse, mse
        integer :: i

        mse = 0.0d0
        do i = 1, n
            mse = mse + (true_vals(i) - est_vals(i))**2
        end do
        rmse = sqrt(mse / n)
    end function calc_rmse

    subroutine calc_stats(times, n, mean_time, std_time)
        integer, intent(in) :: n
        real(dp), intent(in) :: times(n)
        real(dp), intent(out) :: mean_time, std_time
        real(dp) :: var
        integer :: i

        mean_time = sum(times) / n
        var = 0.0d0
        do i = 1, n
            var = var + (times(i) - mean_time)**2
        end do
        std_time = sqrt(var / n)
    end subroutine calc_stats

    subroutine init_random(seed)
        integer, intent(in) :: seed
        integer :: n, i
        integer, allocatable :: seed_arr(:)

        call random_seed(size=n)
        allocate(seed_arr(n))
        do i = 1, n
            seed_arr(i) = seed + i * 37
        end do
        call random_seed(put=seed_arr)
        deallocate(seed_arr)
    end subroutine init_random
    
    subroutine swap_alloc(a, b)
        real(dp), allocatable, intent(inout) :: a(:), b(:)
        real(dp), allocatable :: tmp(:)
    
        call move_alloc(a, tmp)
        call move_alloc(b, a)
        call move_alloc(tmp, b)
    end subroutine swap_alloc

    subroutine randn_vec(x, n)
        integer, intent(in) :: n
        real(dp), intent(out) :: x(n)
    
        integer :: i, m
        real(dp), parameter :: pi = 3.141592653589793d0
        real(dp) :: r
        ! n が奇数でも OK なように (n+1)/2 個生成
        real(dp) :: u1((n+1)/2), u2((n+1)/2)
    
        m = (n + 1) / 2
    
        ! 一括で一様乱数を生成
        call random_number(u1)
        call random_number(u2)
    
        do i = 1, m
            if (u1(i) < 1.0d-10) u1(i) = 1.0d-10
            r = sqrt(-2.0d0 * log(u1(i)))
    
            ! 1 個目
            x(2*i-1) = r * cos(2.0d0 * pi * u2(i))
    
            ! 2 個目（n が奇数なら最後の 1 個はスキップ）
            if (2*i <= n) then
                x(2*i) = r * sin(2.0d0 * pi * u2(i))
            end if
        end do
    end subroutine randn_vec

    subroutine load_data(filename, x1, y1, x2, y2, x3, y3, n)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n
        real(dp), intent(out) :: x1(n), y1(n), x2(n), y2(n), x3(n), y3(n)
        integer :: io_unit, i, t_idx
        character(len=256) :: line

        open(newunit=io_unit, file=filename, status='old')
        read(io_unit, '(A)') line  ! ヘッダースキップ

        do i = 1, n
            read(io_unit, *) t_idx, x1(i), y1(i), x2(i), y2(i), x3(i), y3(i)
        end do

        close(io_unit)
    end subroutine load_data

end program particle_filter_benchmark