#include <vector>
#include <iomanip>
#include <utility>
#include <cmath>
//#include <chrono>
#include <limits>
#include <cstdlib>

#include <iostream>
#include <stdio.h>

#include <omp.h>

#include "utils.hpp"

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: " fmt "\n", ## args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

const double A1 = -3.0;
const double B1 = 3.0;

const double A2 =  0.0;
const double B2 =  2.0;

const double k_lt = 2.0/3.0;
const double b_lt = 2.0;

const double k_rt = -2.0/3.0;
const double b_rt = 2.0;

double h1, h2;
double delta = 1e-6;
double eps;


bool in_D(double x, double y) {
    return (x < 0)? y < (k_lt*x + b_lt):y < (k_rt*x + b_rt);
}

double x_intersec(double y_val, bool right){
    return right ? (- 3.0/2.0 * (y_val - B2)) : (3.0/2.0 * (y_val - B2));
}

double y_intersec(double x_val, bool right){
    return right ? (- 2.0/3.0 * x_val + B2) : (2.0/3.0 * x_val + B2);
}


double a_ij(size_t i, size_t j) {
    if (i==0 or j==0) {
        return 0.0;
    }

    double x_min = A1 + h1 * i - 0.5 * h1;
    double x_max = A1 + h1 * i + 0.5 * h1;

    double y_min = A2 + h2 * j - 0.5 * h2;
    double y_max = A2 + h2 * j + 0.5 * h2;


    if (in_D(x_min, y_min) and in_D(x_min, y_max)) 
        return 1.0;

    if (not in_D(x_min, y_min) and not in_D(x_min, y_max))
        return 1.0 / eps;
    
    double y_inter = y_intersec(x_min, x_min > 0);
    double l = (y_inter - y_min);

    return l/h2 + (1 - l/h2)* (1/eps);
}

double b_ij(size_t i, size_t j) {
    if (i == 0 or j==0) {
        return 0;
    }

    double x_min = A1 + h1 * i - 0.5 * h1;
    double x_max = A1 + h1 * i + 0.5 * h1;

    double y_min = A2 + h2 * j - 0.5 * h2;


    if (in_D(x_min, y_min) and in_D(x_max, y_min)) {
        return 1;
    }

    if (not in_D(x_min, y_min) and not in_D(x_max, y_min)) {
        if (x_min*x_max < 0 ){
            double x_inter_left = x_intersec(y_min,  false);
            double x_inter_right = x_intersec(y_min,  true);
            double l = x_inter_right - x_inter_left;
            return l/h1 + (1 - l/h1) * (1/eps); 
        }
        return 1 / eps;
    }

    double x_inter = x_intersec(y_min,  x_min > 0);
    double l = (x_min < 0) ? (x_max - x_inter): (x_inter - x_min);
    
    return l/h1 + (1 - l/h1)* (1/eps);
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
    bool rb_p =  in_D(x_max, y_min);
    bool rt_p = in_D(x_max, y_max);
    if (lb_p and lt_p and
        rt_p and rb_p) {
        return 1.0;
    } 


    if (not lb_p and not lt_p and
        not rt_p and not rb_p) {

        if (x_max*x_min < 0){
            intersecX_with_ymax = x_intersec(y_max, true);
            intersecX_with_ymin = x_intersec(y_max, false);
            intersecX = x_intersec(y_min, true);
            intersecY = x_intersec(y_min, false);
            S = ((intersecX - intersecY) + (intersecX_with_ymax - intersecX_with_ymin))/2.0 * h2;
            return S / (h1*h2);
        }
        return 0;
    }

    if (lb_p and not lt_p and
        not rt_p and not rb_p){
        intersecX = x_intersec(y_min, true);
        intersecY = y_intersec(x_min, true);
        S = (intersecX - x_min)*(intersecY-y_min)/2.0;
        return S / (h1*h2);
    }

    if (lb_p and lt_p and
        not rt_p and not rb_p){
        intersecX_with_ymin= x_intersec(y_min, true);
        intersecX_with_ymax = x_intersec(y_max, true);
        S = ((intersecX_with_ymax - x_min) + (intersecX_with_ymin - x_min)) / 2.0 * h2;
        return S / (h1*h2);
    }

    if (lb_p and lt_p and
        not rt_p and rb_p){
        intersecX = x_intersec(y_max, true);
        intersecY = y_intersec(x_max, true);
        S = h1*h2 - (x_max - intersecX)*(y_max - intersecY)/2.0;
        return S / (h1*h2);
    }

    if (not lb_p and not lt_p and
        not rt_p and rb_p){
        intersecX = x_intersec(y_min, false);
        intersecY = y_intersec(x_min, false);
        S = (x_max - intersecX)*(intersecY-y_min)/2.0;
        return S / (h1*h2);
    }
    if (not lb_p and not lt_p and
        rt_p and rb_p){
        intersecX_with_ymin= x_intersec(y_min, false);
        intersecX_with_ymax = x_intersec(y_max, false);
        S = ((x_max - intersecX_with_ymax) + (x_max - intersecX_with_ymin)) / 2.0 * h2;
        return S / (h1*h2);
    }
    if (lb_p and not lt_p and
        rt_p and rb_p){
        intersecX = x_intersec(y_max, false);
        intersecY = y_intersec(x_min, false);
        S = h1*h2 - (y_max - intersecY)*(intersecX - x_min)/2.0;
        return S / (h1*h2);
    }
    
    if (lb_p and not lt_p and
        not rt_p and rb_p){
        if (x_min > 0 and x_min < 3){
            intersecY_with_xmin= y_intersec(x_min, true);
            intersecY_with_xmax = y_intersec(x_max, true);
            S = ((intersecY_with_xmax - y_min) + (intersecY_with_xmin - y_min)) / 2.0 * h1;
            return S / (h1*h2);
        }
        else if (x_max > -3 and x_max < 0){
            intersecY_with_xmin= y_intersec(x_min, false);
            intersecY_with_xmax = y_intersec(x_max, false);
            S = ((intersecY_with_xmax - y_min) + (intersecY_with_xmin - y_min)) / 2.0 * h1;
            return S / (h1*h2);           
        }
        else {
            intersecY_with_xmin = y_intersec(x_min, false);
            intersecY_with_xmax = y_intersec(x_max, true);
            double intersecX_with_ymax_right = x_intersec(y_max, true);
            double intersecX_with_ymax_left = x_intersec(y_max, false);
            S = h1*h2 - (y_max - intersecY_with_xmax)*(x_max - intersecX_with_ymax_right)/2.0 - (intersecX_with_ymax_left - x_min)*(y_max - intersecY_with_xmin)/2.0;
            return S / (h1*h2);

        }
        double intersecY_with_ymin= y_intersec(x_min, true);
        double intersecY_with_ymax = y_intersec(x_max, true);
        S = ((intersecY_with_ymax - y_min) + (intersecY_with_ymin - y_min)) / 2.0 * h1;
        return S / (h1*h2);
    }

    return -100;
}

void R (std::vector<double> &res, const std::vector<double> &w, bool flag, size_t M, size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            res[i*(N+1) + j] = 
                (-1.0 / h1)  * (
                    a_ij(i + 1, j) * (w[(i+1)*(N+1) + j] - w[i    *(N+1) + j]) / h1 -
                    a_ij(i, j)     * (w[i    *(N+1) + j] - w[(i-1)*(N+1) + j]) / h1
                )  +
                (-1.0 / h2) * (
                    b_ij(i, j + 1) * (w[i*(N+1) + j+1] - w[i*(N+1) +   j]) / h2 -
                    b_ij(i, j)     * (w[i*(N+1) +   j] - w[i*(N+1) + j-1]) / h2
                );
        }
    }
    if (flag) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i < M; ++i) {
            for (size_t j = 1; j < N; ++j) {
                double tmp = f_ij(i, j);
                res[i*(N+1) + j] -= tmp;
                
            }
        }
    }
}

int main (int argc, char **argv) {
    std::cout << std::setprecision(15);
    size_t M;
    size_t N;
    double delta = 1e-6;
    if (argc > 2){
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    else{
        printf("worng arg count %d", argc);
        return 1;
    }
    if (argc == 4) {    
        delta= atof(argv[3]); 
    }
    else if (argc == 3){   
        if (N*M>6000)
            delta = 2*1e-7;
        if (N*M>10000)
            delta = 1.8 * 1e-8;
    }   
    else{
        printf("worng arg count %d", argc);
        return 1;
    }

    h1 = (B1 - A1) / M;
    h2 = (B2 - A2) / N;
    eps = (h1>h2)?h2:h1;
    eps *= eps;


    std::vector<double> Ar((M+1)*(N+1), 0);
    std::vector<double> w((M+1)*(N+1), 0);
    std::vector<double> r((M+1)*(N+1), 0);
    
    int it = 0;
    double error_rate = 0, tau = 1;
    double start_time = omp_get_wtime();
    double dot_r = 0;
    do {

        R(r, w, true, M, N); 
        R(Ar, r, false, M, N);
        dot_r = dot(r, r, M,  N, h1, h2);
        tau = (dot_r / dot(Ar, r, M,  N, h1,  h2));
        new_w(w, r, tau, M, N);

        error_rate = sqrt(tau * tau * dot_r);
        #ifdef DEBUG
        if (it % 10000 == 0){
            DEBUG_PRINT("it - %d\n", it);
            print_matrix(r, M, N);
            print_matrix(Ar, M, N);

            DEBUG_PRINT("error_rate %f\ntau %f\n", error_rate,  1/tau);
            DEBUG_PRINT("r %f\nAr@r %f\n", dot(r, r, M,  N, h1,  h2),  dot(Ar, r, M,  N, h1,  h2));
            DEBUG_PRINT("dt %f\neps %f\n\n", delta, eps);
        }
        #endif
        it += 1;
    } while(error_rate > delta);

    std::cout << "time: " <<  omp_get_wtime() - start_time << '\n';;
    std::cout << "delta: " << delta << '\n';
    std::cout << "size: " << M << 'x' << N << '\n';
    std::cout << "it: " << it << '\n';
    return 0;
}