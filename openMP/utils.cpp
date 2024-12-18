#include "utils.hpp"

void print_matrix(std::vector<double> v, size_t M, size_t N) {
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            printf("%f ", v[i*(N+1) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void new_w(std::vector<double> &w, const std::vector<double> &r, double tau, size_t M, size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            w[i*(N+1) + j] -= tau * r[i*(N+1) + j];
        }
    }
}

double dot(const std::vector<double> &a, const std::vector<double> &b, size_t M, size_t N, double h1, double h2) {
    double sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for (int i = 1; i < M; i++) {
        for (int j = 1; j < N; j++) {
            sum += a[i*(N+1) + j] * b[i*(N+1) + j];
        }
    } 
    return sum * h1 * h2;
}