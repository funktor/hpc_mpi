#include <immintrin.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <deque>
#include <tuple>
#include <map>
#include <fcntl.h>
#include <functional>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <ctime> 
#include <stdbool.h>    // bool type

using namespace std;

void generate(double *inp, int n, int m){
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            inp[i*m+j] = dist(engine);
        }
    }
}

void print_arr(const double *arr, const int n, const int m) {
    std::cout << "[";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << arr[i*m+j] << ",";
        }
    }
    std::cout << "]";

    std::cout << std::endl;
    std::cout << std::endl;
}

double *dot_simd(const double *a, const double *b, const int n, const int m, const int p) {
    double *out = new double[n*p];
    
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            __m512d c = _mm512_set1_pd(a[i*m+k]);
            for (int j = 0; j < p; j += 8) {
                if (j+8 > p) {
                    for (int h = j; h < p; h++) {
                        out[i*p+h] += a[i*m+k]*b[k*p+h];
                    }
                }
                else {
                    __m512d x = _mm512_loadu_pd(&b[k*p+j]);
                    __m512d y = _mm512_loadu_pd(&out[i*p+j]);
                    x = _mm512_mul_pd(x, c);
                    y = _mm512_add_pd(y, x);
                    _mm512_storeu_pd(&out[i*p+j], y);
                }
            }
        }
    }

    return out;
}

double *dot(const double *a, const double *b, const int n, const int m, const int p) {
    double *out = new double[n*p];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) {
                out[i*p+k] += a[i*m+j]*b[j*p+k];
            }
        }
    }

    return out;
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int p = atoi(argv[3]);

    double *a, *b;
    a = new double[n*m];
    b = new double[m*p];

    generate(a, n, m);
    generate(b, m, p);

    auto start = std::chrono::high_resolution_clock::now();
    double *out = dot_simd(a, b, n, m, p);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    // print_arr(out, n, p);
    std::cout << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    out = dot(a, b, n, m, p);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    // print_arr(out, n, p);
    std::cout << duration.count() << std::endl;

    return 0;
}