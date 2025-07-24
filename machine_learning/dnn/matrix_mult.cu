#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
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
#include <fstream>
#include <cmath>
#include <variant>
#include <omp.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_ERR 1e-10
#define TILE_WIDTH 32
#define COARSE_FACTOR 8

bool are_equal(float *x, float *y, int start, int end) {
    for (int i = start; i < end; i++) {
        if (fabs(x[i]-y[i]) > 0.1) {
            std::cout << i << " " << x[i] << " " << y[i] << std::endl;
            return false;
        }
    }
    return true;
}

void print_vector(float *x, int start, int end) {
    std::cout << "[";
    for (int i = start; i <= end; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

void generate_data(float *x, int n, int m) {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < n*m; i++) x[i] = dist(rng);
}

__global__ 
void cuda_mul(float *a, float *b, float *c, int n, int m, int p) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        float res = 0.0;
        for (int i = 0; i < m; i++) res += a[row*m+i]*b[i*p+col];
        c[row*p+col] = res;
    }
}

__global__ 
void cuda_mul_bt(float *a, float *b, float *c, int n, int m, int p) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        float res = 0.0;
        for (int i = 0; i < m; i++) res += a[row*m+i]*b[col*m+i];
        c[row*p+col] = res;
    }
}

__global__ 
void cuda_mul_tiled(float *a, float *b, float *c, int n, int m, int p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    float res = 0.0;
    for (int ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[ty*TILE_WIDTH+tx] = b[(ph*TILE_WIDTH+ty)*p + col];
        else Nds[ty*TILE_WIDTH+tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) res += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
        __syncthreads();
    }

    if (row < n && col < p) c[row*p+col] = res; 
}

__global__ 
void cuda_mul_bt_tiled(float *a, float *b, float *c, int n, int m, int p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    float res = 0.0;
    for (int ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[tx*TILE_WIDTH+ty] = b[col*m + (ph*TILE_WIDTH+ty)];
        else Nds[tx*TILE_WIDTH+ty] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) res += Mds[ty*TILE_WIDTH+i]*Nds[tx*TILE_WIDTH+i];
        __syncthreads();
    }

    if (row < n && col < p) c[row*p+col] = res;
}

__global__ 
void cuda_mul_bt_tiled_coarsened(float *a, float *b, float *c, int n, int m, int p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    float Pval[COARSE_FACTOR];
    for (int r = 0; r < COARSE_FACTOR; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        for (int r = 0; r < COARSE_FACTOR; r++) {
            int col = col_start + r*TILE_WIDTH;

            if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[tx*TILE_WIDTH+ty] = b[col*m + (ph*TILE_WIDTH+ty)];
            else Nds[tx*TILE_WIDTH+ty] = 0.0f;
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) Pval[r] += Mds[ty*TILE_WIDTH+i]*Nds[tx*TILE_WIDTH+i];
            __syncthreads();
        }
    }

    for (int r = 0; r < COARSE_FACTOR; r++) {
        int col = col_start + r*TILE_WIDTH;
        if (row < n && col < p) c[row*p+col] = Pval[r];
    }
}

void mat_mul(float *a, float *b, float *c, int n, int m, int p) {
    omp_set_num_threads(8);

    #pragma omp parallel for shared(c)
    for (int i = 0; i < n*p; i++) c[i] = 0.0;

    #pragma omp parallel for shared(a, b, c)
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) c[i*p+k] += a[i*m+j]*b[j*p+k];
        }
    }
}

void mat_mul_bt(float *a, float *b, float *c, int n, int m, int p) {
    omp_set_num_threads(8);

    #pragma omp parallel for shared(c)
    for (int i = 0; i < n*p; i++) c[i] = 0.0;

    #pragma omp parallel for shared(a, b, c)
    for(int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < m; j++) c[i*p+k] += a[i*m+j]*b[k*m+j];
        }
    }
}

int main(){
    int n = 3195;
    int m = 567;
    int p = 1872;

    float *a, *b, *bt, *c1, *c2, *c3, *c4;

    size_t size_a = sizeof(float)*n*m;
    size_t size_b = sizeof(float)*m*p;
    size_t size_c = sizeof(float)*n*p;

    cudaMallocManaged(&a, size_a);
    cudaMallocManaged(&b, size_b);
    cudaMallocManaged(&bt, size_b);
    cudaMallocManaged(&c1, size_c);
    cudaMallocManaged(&c2, size_c);
    cudaMallocManaged(&c3, size_c);
    cudaMallocManaged(&c4, size_c);
    // d = (float*)malloc(size_c);

    generate_data(a, n, m);
    generate_data(b, m, p);
    generate_data(bt, p, m);

    mat_mul_bt(a, bt, c1, n, m, p);

    dim3 bd(32, 32, 1);
    dim3 gd(ceil(p/32.0), ceil(n/32.0), 1);

    dim3 bd1(32, 32, 1);
    dim3 gd1(ceil(p/(COARSE_FACTOR*32.0)), ceil(n/32.0), 1);

    auto start = std::chrono::high_resolution_clock::now();

    cuda_mul_bt<<<gd, bd>>>(a, bt, c2, n, m, p);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c1, c2, 0, n*p) << std::endl;

    // print_vector(c, 5000, 5100);

    start = std::chrono::high_resolution_clock::now();

    cuda_mul_bt_tiled<<<gd, bd>>>(a, bt, c3, n, m, p);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c1, c3, 0, n*p) << std::endl;


    start = std::chrono::high_resolution_clock::now();

    cuda_mul_bt_tiled_coarsened<<<gd1, bd1>>>(a, bt, c4, n, m, p);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c1, c4, 0, n*p) << std::endl;

    // print_vector(c, 5000, 5100);

    cudaFree(a);
    cudaFree(b);
    cudaFree(bt);
    cudaFree(c1);
    cudaFree(c2);
    cudaFree(c3);
    cudaFree(c4);
    // free(d);
}
