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

#define BLOCK_WIDTH 1024
#define MAX_N 1e7
__device__ int counter=0;

bool are_equal(float *x, float *y, int start, int end) {
    for (int i = start; i < end; i++) {
        if (fabs(x[i]-y[i])/fabs(x[i]) > 0.01) {
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

void prefix_sum(float *arr, float *out, int n) {
    for (int i = 0; i < n; i++) {
        if (i == 0) out[i] = arr[i];
        else out[i] = out[i-1] + arr[i];
    }
}

__device__
void prefix_sum_kogge_stone_block(float *arr, float *out, int n) {
    __shared__ float XY[BLOCK_WIDTH];
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n) XY[threadIdx.x] = arr[index]; 
    else XY[threadIdx.x] = 0.0f;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = XY[threadIdx.x-stride];
        __syncthreads();
        XY[threadIdx.x] += temp;
        __syncthreads();
    }

    if (index < n) out[index] = XY[threadIdx.x];
}

__device__
void prefix_sum_kogge_stone_block_flagged(float *arr, float *out, int *flags, int n) {
    __shared__ float XY[BLOCK_WIDTH];
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    while (flags[index] != 1) {}

    if (index < n) XY[threadIdx.x] = arr[index]; 
    else XY[threadIdx.x] = 0.0f;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = XY[threadIdx.x-stride];
        __syncthreads();
        XY[threadIdx.x] += temp;
        __syncthreads();
    }

    if (index < n) out[index] = XY[threadIdx.x];
}

__global__
void prefix_sum_kogge_stone_p1(float *arr, float *out, int *flags, float *S, float *S_out, int n, int m) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    prefix_sum_kogge_stone_block(arr, out, n);
    __syncthreads();
    if (blockIdx.x < m && threadIdx.x == 0) {
        S[blockIdx.x] = out[min((blockIdx.x+1)*blockDim.x-1, n-1)];
        __threadfence();
        flags[blockIdx.x] = 1;
    }
    prefix_sum_kogge_stone_block_flagged(S, S_out, flags, m);
    __syncthreads();
    if (index < n && blockIdx.x > 0) out[index] += S_out[blockIdx.x-1];
}

__global__
void prefix_sum_kogge_stone_p2(float *S, float *S_out, int m) {
    prefix_sum_kogge_stone_block(S, S_out, m);
}

__global__
void prefix_sum_kogge_stone_update(float *out, float *S_out, int n, int m) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n && blockIdx.x > 0) out[index] += S_out[blockIdx.x-1];
}

int main(){
    int n = 4096;
    int m = int(ceil(float(n)/BLOCK_WIDTH));

    float *a, *S, *S_out, *c1, *c2;

    cudaMallocManaged(&a, n*sizeof(float));
    cudaMallocManaged(&c1, n*sizeof(float));
    cudaMallocManaged(&c2, n*sizeof(float));
    cudaMallocManaged(&S, m*sizeof(float));
    cudaMallocManaged(&S_out, m*sizeof(float));

    for (int i = 0; i < m; i++) S[i] = 0.0;
    for (int i = 0; i < m; i++) S_out[i] = 0.0;

    generate_data(a, n, 1);

    auto start = std::chrono::high_resolution_clock::now();
    prefix_sum(a, c1, n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;

    // print_vector(c1, 2000, 2100);

    start = std::chrono::high_resolution_clock::now();

    prefix_sum_kogge_stone_p1<<<m, BLOCK_WIDTH>>>(a, c2, S, S_out, n, m);
    cudaDeviceSynchronize();
    // prefix_sum_kogge_stone_p2<<<m, BLOCK_WIDTH>>>(S, S_out, m);
    // cudaDeviceSynchronize();
    // prefix_sum_kogge_stone_update<<<m, BLOCK_WIDTH>>>(c2, S_out, n, m);
    // cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    // print_vector(c2, 2000, 2100);
    std::cout << are_equal(c2, c1, 0, n) << std::endl;
}

// 5, 3     0 1 2 3 4  -> 0, 0 1, 0 1 2, 1 2 3, 2 3 4, 3 4, 4