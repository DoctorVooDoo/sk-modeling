#include <stdio.h>
#include <vector>
#include <omp.h>


void print_matrix(std::vector<double> v, size_t M, size_t N);
void new_w(std::vector<double> &w, const std::vector<double> &r, double tau, size_t M, size_t N);
double dot(const std::vector<double> &a, const std::vector<double> &b, size_t M, size_t N, double h1, double h2);