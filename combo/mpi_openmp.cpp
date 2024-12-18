#include <vector>
#include <iomanip>
#include <utility>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: " fmt "\n", ## args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

#define FILLER -100

const double A1 = -3.0, B1 = 3.0, A2 = 0.0, B2 = 2.0;
const double k_lt = 2.0 / 3.0, b_lt = 2.0, k_rt = -2.0 / 3.0, b_rt = 2.0;
const double delta = 1e-6; 
double eps;

MPI_Comm comm;
int rank, size, dim[2] = {0, 0}, coords[2] = {0, 0}, period[2] = {0, 0};
int x_left, y_botn;
double h1, h2;

double *right_buf, *left_buf, *top_buf, *botn_buf, *tmp_buf;

void print_matrix(const std::vector<double>& v, size_t M, size_t N) {
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            printf("%f ", v[i * (N + 1) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool in_D(double x, double y) {
    return (x < 0) ? (y < (k_lt * x + b_lt)) : (y < (k_rt * x + b_rt));
}

double x_intersec(double y_val, bool right) {
    return right ? (-3.0 / 2.0 * (y_val - B2)) : (3.0 / 2.0 * (y_val - B2));
}

double y_intersec(double x_val, bool right) {
    return right ? (-2.0 / 3.0 * x_val + B2) : (2.0 / 3.0 * x_val + B2);
}

double a_ij(size_t i, size_t j) {
    if (i == 0 || j == 0) return 0.0;

    double x_min = A1 + h1 * i - 0.5 * h1, x_max = A1 + h1 * i + 0.5 * h1;
    double y_min = A2 + h2 * j - 0.5 * h2, y_max = A2 + h2 * j + 0.5 * h2;

    if (in_D(x_min, y_min) && in_D(x_min, y_max)) return 1.0;
    if (!in_D(x_min, y_min) && !in_D(x_min, y_max)) return 1.0 / eps;

    double y_inter = y_intersec(x_min, x_min > 0);
    double l = (y_inter - y_min);

    return l / h2 + (1 - l / h2) * (1 / eps);
}

double b_ij(size_t i, size_t j) {
    if (i == 0 || j == 0) return 0;

    double x_min = A1 + h1 * i - 0.5 * h1, x_max = A1 + h1 * i + 0.5 * h1;
    double y_min = A2 + h2 * j - 0.5 * h2;

    if (in_D(x_min, y_min) && in_D(x_max, y_min)) return 1;
    if (!in_D(x_min, y_min) && !in_D(x_max, y_min)) {
        if (x_min * x_max < 0) {
            double x_inter_left = x_intersec(y_min, false);
            double x_inter_right = x_intersec(y_min, true);
            double l = x_inter_right - x_inter_left;
            return l / h1 + (1 - l / h1) * (1 / eps);
        }
        return 1 / eps;
    }

    double x_inter = x_intersec(y_min, x_min > 0);
    double l = (x_min < 0) ? (x_max - x_inter) : (x_inter - x_min);

    return l / h1 + (1 - l / h1) * (1 / eps);
}

double f_ij(size_t i, size_t j) {
    double x_min = A1 + h1 * i - 0.5 * h1;
    double x_max = A1 + h1 * i + 0.5 * h1;
    double y_min = A2 + h2 * j - 0.5 * h2;
    double y_max = A2 + h2 * j + 0.5 * h2;

    double intersecY_with_xmin, intersecY_with_xmax, intersecY, S;
    double intersecX_with_ymax, intersecX_with_ymin, intersecX;

    bool lb_p = in_D(x_min, y_min);
    bool lt_p = in_D(x_min, y_max);
    bool rb_p = in_D(x_max, y_min);
    bool rt_p = in_D(x_max, y_max);

    if (lb_p && lt_p && rt_p && rb_p) {
        return 1.0;
    }

    if (!lb_p && !lt_p && !rt_p && !rb_p) {
        if (x_max * x_min < 0) {
            intersecX_with_ymax = x_intersec(y_max, true);
            intersecX_with_ymin = x_intersec(y_min, false);
            intersecX = x_intersec(y_min, true);
            intersecY = x_intersec(y_max, false);
            S = ((intersecX - intersecY) + (intersecX_with_ymax - intersecX_with_ymin)) / 2.0 * h2;
            return S / (h1 * h2);
        }
        return 0;
    }

    if (lb_p && !lt_p && !rt_p && !rb_p) {
        intersecX = x_intersec(y_min, true);
        intersecY = y_intersec(x_min, true);
        S = (intersecX - x_min) * (intersecY - y_min) / 2.0;
        return S / (h1 * h2);
    }

    if (lb_p && lt_p && !rt_p && !rb_p) {
        intersecX_with_ymin = x_intersec(y_min, true);
        intersecX_with_ymax = x_intersec(y_max, true);
        S = ((intersecX_with_ymax - x_min) + (intersecX_with_ymin - x_min)) / 2.0 * h2;
        return S / (h1 * h2);
    }

    if (lb_p && lt_p && !rt_p && rb_p) {
        intersecX = x_intersec(y_max, true);
        intersecY = y_intersec(x_max, true);
        S = h1 * h2 - (x_max - intersecX) * (y_max - intersecY) / 2.0;
        return S / (h1 * h2);
    }

    if (!lb_p && !lt_p && rt_p && rb_p) {
        intersecX_with_ymin = x_intersec(y_min, false);
        intersecX_with_ymax = x_intersec(y_max, false);
        S = ((x_max - intersecX_with_ymax) + (x_max - intersecX_with_ymin)) / 2.0 * h2;
        return S / (h1 * h2);
    }

    if (lb_p && !lt_p && rt_p && rb_p) {
        intersecX = x_intersec(y_max, false);
        intersecY = y_intersec(x_min, false);
        S = h1 * h2 - (y_max - intersecY) * (intersecX - x_min) / 2.0;
        return S / (h1 * h2);
    }

    if (lb_p && !lt_p && !rt_p && rb_p) {
        if (x_min > 0 && x_min < 3) {
            intersecY_with_xmin = y_intersec(x_min, true);
            intersecY_with_xmax = y_intersec(x_max, true);
            S = ((intersecY_with_xmax - y_min) + (intersecY_with_xmin - y_min)) / 2.0 * h1;
            return S / (h1 * h2);
        } else if (x_max > -3 && x_max < 0) {
            intersecY_with_xmin = y_intersec(x_min, false);
            intersecY_with_xmax = y_intersec(x_max, false);
            S = ((intersecY_with_xmax - y_min) + (intersecY_with_xmin - y_min)) / 2.0 * h1;
            return S / (h1 * h2);
        } else {
            intersecY_with_xmin = y_intersec(x_min, false);
            intersecY_with_xmax = y_intersec(x_max, true);
            double intersecX_with_ymax_right = x_intersec(y_max, true);
            double intersecX_with_ymax_left = x_intersec(y_max, false);
            S = h1 * h2 - (y_max - intersecY_with_xmax) * (x_max - intersecX_with_ymax_right) / 2.0
                - (intersecX_with_ymax_left - x_min) * (y_max - intersecY_with_xmin) / 2.0;
            return S / (h1 * h2);
        }
    }

    return -100;
}

void new_w(std::vector<double>& w, const std::vector<double>& r, double tau, size_t M, size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i <= M; ++i) {
        for (size_t j = 1; j <= N; ++j) {
            size_t index = i * (N + 2) + j;
            w[index] -= tau * r[index];
        }
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b, size_t M, size_t N, double h1, double h2) {
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 1; i <= M; ++i) {
        for (size_t j = 1; j <= N; ++j) {
            size_t index = i * (N + 2) + j;
            sum += a[index] * b[index];
        }
    }
    sum *= h1 * h2;

    double global_sum = 0;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

void R(std::vector<double>& res, std::vector<double>& w, bool flag, size_t col_size, size_t row_size) {
    std::vector<double> left_buf(row_size + 2), right_buf(row_size + 2);
    std::vector<double> botn_buf(col_size + 2), top_buf(col_size + 2);
    std::vector<double> tmp_buf(std::max(col_size, row_size) + 2, FILLER);

    int rank_left, rank_right, rank_up, rank_down;
    MPI_Status status;

    MPI_Cart_shift(comm, 0, 1, &rank_left, &rank_right); 
    MPI_Cart_shift(comm, 1, 1, &rank_down, &rank_up); 

    if (rank_left >= 0) {
        MPI_Sendrecv(left_buf.data(), row_size + 2, MPI_DOUBLE, rank_left, 1,
                     tmp_buf.data(), row_size + 2, MPI_DOUBLE, rank_left, 1, comm, &status);
        std::copy(tmp_buf.begin(), tmp_buf.begin() + row_size, w.begin() + 1);
    }

    if (rank_right >= 0) { 
        MPI_Sendrecv(right_buf.data(), row_size + 2, MPI_DOUBLE, rank_right, 1,
                     tmp_buf.data(), row_size + 2, MPI_DOUBLE, rank_right, 1, comm, &status);
        std::copy(tmp_buf.begin(), tmp_buf.begin() + row_size, w.begin() + col_size * (row_size + 2) + 1);
    }

    if (rank_down >= 0) { 
        MPI_Sendrecv(botn_buf.data(), col_size + 2, MPI_DOUBLE, rank_down, 1,
                     tmp_buf.data(), col_size + 2, MPI_DOUBLE, rank_down, 1, comm, &status);
        for (size_t i = 0; i < col_size; ++i)
            w[i * (row_size + 2)] = tmp_buf[i];
    }

    if (rank_up >= 0) { 
        MPI_Sendrecv(top_buf.data(), col_size + 2, MPI_DOUBLE, rank_up, 1,
                     tmp_buf.data(), col_size + 2, MPI_DOUBLE, rank_up, 1, comm, &status);
        for (size_t i = 0; i < col_size; ++i)
            w[i * (row_size + 2) + row_size + 1] = tmp_buf[i];
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i <= col_size; ++i) {
        for (size_t j = 1; j <= row_size; ++j) {
            size_t idx = i * (row_size + 2) + j;
            res[idx] = (-1.0 / h1) * (
                a_ij(i + x_left - 1 + 1, j + y_botn - 1) * (w[(i + 1) * (row_size + 2) + j] - w[i * (row_size + 2) + j]) / h1 -
                a_ij(i + x_left - 1, j + y_botn - 1) * (w[i * (row_size + 2) + j] - w[(i - 1) * (row_size + 2) + j]) / h1
            ) + (-1.0 / h2) * (
                b_ij(i + x_left - 1, j + y_botn - 1 + 1) * (w[i * (row_size + 2) + j + 1] - w[i * (row_size + 2) + j]) / h2 -
                b_ij(i + x_left - 1, j + y_botn - 1) * (w[i * (row_size + 2) + j] - w[i * (row_size + 2) + j - 1]) / h2
            );
        }
    }

    if (flag) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i <= col_size; ++i) {
            for (size_t j = 1; j <= row_size; ++j) {
                size_t idx = i * (row_size + 2) + j;
                res[idx] -= f_ij(i + x_left - 1, j + y_botn - 1);
            }
        }
    }
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(15);

    int M, N;
    double delta = 1e-6;
    int terminate_signal = 0;
    int col_size, row_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Dims_create(size, 2, dim);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);

    if (argc > 2) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    } else {
        std::cerr << "Wrong argument count " << argc << "\n";
        terminate_signal = 1;
    }

    if (argc == 4) {
        delta = std::atof(argv[3]);
    } else if (argc == 3) {
        if (N * M > 6000) delta = 2 * 1e-7;
        if (N * M > 10000) delta = 2.25 * 1e-8;
    }

    if (terminate_signal == 1) {
        std::cerr << "Process " << rank << " received termination signal\n";
        MPI_Finalize();
        return 1;
    }

    h1 = (B1 - A1) / M;
    h2 = (B2 - A2) / N;
    eps = (h1 > h2) ? h2 : h1;
    eps *= eps;

    col_size = (coords[0] >= (M - 1) % dim[0]) ? (M - 1) / dim[0] : (M - 1) / dim[0] + 1;
    row_size = (coords[1] >= (N - 1) % dim[1]) ? (N - 1) / dim[1] : (N - 1) / dim[1] + 1;

    x_left = (coords[0] < (M - 1) % dim[0]) ? 1 + coords[0] * col_size : 1 + coords[0] * col_size + (M - 1) % dim[0];
    y_botn = (coords[1] < (N - 1) % dim[1]) ? 1 + coords[1] * row_size : 1 + coords[1] * row_size + (N - 1) % dim[1];

    std::vector<double> w((1 + col_size + 1) * (1 + row_size + 1), 0);
    std::vector<double> r((1 + col_size + 1) * (1 + row_size + 1), 0);
    std::vector<double> Ar((1 + col_size + 1) * (1 + row_size + 1), 0);

    int it = 0;
    double error_rate = 0, tau = 1;
    double start_time = omp_get_wtime();

    do {
        R(r, w, true, col_size, row_size);
        R(Ar, r, false, col_size, row_size);
        double dot_r = dot(r, r, col_size, row_size, h1, h2);
        tau = (dot_r / dot(Ar, r, col_size, row_size, h1, h2));
        new_w(w, r, tau, col_size, row_size);

        error_rate = std::sqrt(tau * tau * dot_r);

        #ifdef DEBUG
        if (it % 10000 == 0) {
            DEBUG_PRINT("it - %d\n", it);
            print_matrix(r, M, N);
            print_matrix(Ar, M, N);

            DEBUG_PRINT("error_rate %f\ntau %f\n", error_rate, 1 / tau);
            DEBUG_PRINT("r %f\nAr@r %f\n", dot(r, r, M, N, h1, h2), dot(Ar, r, M, N, h1, h2));
            DEBUG_PRINT("dt %f\neps %f\n\n", delta, eps);
        }
        #endif

        it++;
    } while (error_rate > delta);

    if (rank == 0) {
        std::cout << "time: " << omp_get_wtime() - start_time << '\n';
        std::cout << "delta: " << delta << '\n';
        std::cout << "size: " << M << 'x' << N << '\n';
        std::cout << "it: " << it << '\n';
    }

    MPI_Finalize();
    return 0;
}
