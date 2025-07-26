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
#define COARSE_FACTOR 4
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
void prefix_sum_kogge_stone_block(float *arr, float *XY, int n) {
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
}

__device__
void prefix_sum_brent_kung_block(float *arr, float *XY, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n) XY[threadIdx.x] = arr[index]; 
    else XY[threadIdx.x] = 0.0f;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int i = 2*(threadIdx.x+1)*stride-1;
        if (i < BLOCK_WIDTH && i >= stride) XY[i] += XY[i-stride];
        __syncthreads();
    }

    for (unsigned int stride = BLOCK_WIDTH/4; stride > 0; stride /= 2) {
        int i = 2*(threadIdx.x+1)*stride-1;
        if (i + stride < BLOCK_WIDTH) XY[i + stride] += XY[i];
        __syncthreads();
    }
}

__device__
void prefix_sum_brent_kung_block_coarsened(float *arr, float *XY, int n) {
    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n) XY[i] = arr[index]; 
        else XY[i] = 0.0f;
    }

    __syncthreads();

    for (unsigned int stride = 1; stride < COARSE_FACTOR*blockDim.x; stride *= 2) {
        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            int j = 2*(i+1)*stride-1;
            if (j < COARSE_FACTOR*BLOCK_WIDTH && j >= stride) XY[j] += XY[j-stride];
        }
        __syncthreads();
    }

    for (unsigned int stride = COARSE_FACTOR*BLOCK_WIDTH/4; stride > 0; stride /= 2) {
        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            int j = 2*(i+1)*stride-1;
            if (j + stride < COARSE_FACTOR*BLOCK_WIDTH) XY[j + stride] += XY[j];
        }
        __syncthreads();
    }
}

// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  0 - 0+1 -> 1
//   1   3   5   7   9    11    13    15  1 - 2+3 -> 3 
//       3       7        11          15  2 - 4+5 -> 5
//               7                    15
//                                    15
//                        11              0 - 7 + 11 -> 11
//           5       9          13        0 - 3 + 5 -> 5, 1 - 7 + 9 -> 9
//     2   4   6   8   10    12    14

__global__
void prefix_sum(float *arr, float *out, int *flags, float *S, int n, int m) {
    extern __shared__ float XY[];
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    prefix_sum_brent_kung_block(arr, XY, n);

    if (blockIdx.x + 1 < m && threadIdx.x == 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}
        S[blockIdx.x + 1] = S[blockIdx.x] + XY[min(blockDim.x-1, n-1-blockIdx.x*blockDim.x)];
        __threadfence();
        atomicAdd(&flags[blockIdx.x + 1], 1);
    }
    __syncthreads();

    if (blockIdx.x < m && index < n && blockIdx.x > 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}
        XY[threadIdx.x] += S[blockIdx.x];
    }

    if (index < n) out[index] = XY[threadIdx.x];
}

__global__
void prefix_sum_coarsened(float *arr, float *out, int *flags, float *S, int n, int m) {
    extern __shared__ float XY[];

    prefix_sum_brent_kung_block_coarsened(arr, XY, n);

    if (blockIdx.x + 1 < m && threadIdx.x == 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}
        S[blockIdx.x + 1] = S[blockIdx.x] + XY[min(COARSE_FACTOR*blockDim.x-1, n-1-COARSE_FACTOR*blockIdx.x*blockDim.x)];
        __threadfence();
        atomicAdd(&flags[blockIdx.x + 1], 1);
    }
    __syncthreads();

    if (blockIdx.x < m && blockIdx.x > 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}

        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            XY[i] += S[blockIdx.x];
        }
    }

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n) out[index] = XY[i];
    }
}

__global__
void prefix_sum_coarsened_static(float *arr, float *out, int *flags, float *S, int n, int m) {
    __shared__ float XY[COARSE_FACTOR*BLOCK_WIDTH];

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n) XY[i] = arr[index]; 
        else XY[i] = 0.0f;
    }

    __syncthreads();

    for (unsigned int stride = 1; stride < COARSE_FACTOR*blockDim.x; stride *= 2) {
        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            int j = 2*(i+1)*stride-1;
            if (j < COARSE_FACTOR*BLOCK_WIDTH && j >= stride) XY[j] += XY[j-stride];
        }
        __syncthreads();
    }

    for (unsigned int stride = COARSE_FACTOR*BLOCK_WIDTH/4; stride > 0; stride /= 2) {
        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            int j = 2*(i+1)*stride-1;
            if (j + stride < COARSE_FACTOR*BLOCK_WIDTH) XY[j + stride] += XY[j];
        }
        __syncthreads();
    }

    if (blockIdx.x + 1 < m && threadIdx.x == 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}
        S[blockIdx.x + 1] = S[blockIdx.x] + XY[min(COARSE_FACTOR*blockDim.x-1, n-1-COARSE_FACTOR*blockIdx.x*blockDim.x)];
        __threadfence();
        atomicAdd(&flags[blockIdx.x + 1], 1);
    }
    __syncthreads();

    if (blockIdx.x < m && blockIdx.x > 0) {
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}

        for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
            XY[i] += S[blockIdx.x];
        }
    }

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n) out[index] = XY[i];
    }
}

int main(){
    int n = 1e7;
    int m = int(ceil(float(n)/BLOCK_WIDTH));

    float *a, *S, *c1, *c2;
    int *flags;

    cudaMallocManaged(&a, n*sizeof(float));
    cudaMallocManaged(&c1, n*sizeof(float));
    cudaMallocManaged(&c2, n*sizeof(float));
    cudaMallocManaged(&S, m*sizeof(float));
    cudaMallocManaged(&flags, m*sizeof(int));

    for (int i = 0; i < m; i++) S[i] = 0.0;
    for (int i = 0; i < m; i++) flags[i] = 0;
    flags[0] = 1;

    generate_data(a, n, 1);

    auto start = std::chrono::high_resolution_clock::now();
    prefix_sum(a, c1, n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;

    // print_vector(c1, 2000, 2100);

    start = std::chrono::high_resolution_clock::now();

    prefix_sum_coarsened<<<m, BLOCK_WIDTH, BLOCK_WIDTH*COARSE_FACTOR*sizeof(float)>>>(a, c2, flags, S, n, m);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    // print_vector(c2, 2000, 2100);
    std::cout << are_equal(c2, c1, 0, n) << std::endl;
}

// 5, 3     0 1 2 3 4  -> 0, 0 1, 0 1 2, 1 2 3, 2 3 4, 3 4, 4