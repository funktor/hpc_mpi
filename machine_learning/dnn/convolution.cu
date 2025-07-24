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

#define K 11
#define MAX_ERR 1e-10
#define TILE_WIDTH 32
#define OUT_TILE_WIDTH 32
#define INP_TILE_WIDTH (OUT_TILE_WIDTH + (K-1))
__constant__ float F_c[K*K];


bool are_equal(float *x, float *y, int start, int end) {
    for (int i = start; i < end; i++) {
        if (fabs(x[i]-y[i]) > 0.01) {
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

void conv2d(float *a, float *f, float *c, int k, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float res = 0.0;
            for (int u = 0; u < k; u++) {
                for (int v = 0; v < k; v++) {
                    int p = i-(k-1)/2+u;
                    int q = j-(k-1)/2+v;

                    if (p >= 0 && p < n && q >= 0 && q < m) res += a[p*m+q]*f[u*k+v];
                }
            }
            c[i*m+j] = res;
        }
    }
}

__global__ 
void conv2D_basic(float *a, float *f, float *c, int k, int n, int m) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    float res = 0.0f;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            int u = row-(k-1)/2+i;
            int v = col-(k-1)/2+j;

            // check the boundaries
            if (u >= 0 && u < n && v >= 0 && v < m) res += a[u*m+v]*f[i*k+j];
        }
    }

    if (row < n && col < m) c[row*m+col] = res;
}

__global__ 
void conv2D_constant_mem(float *a, float *c, int n, int m) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    float res = 0.0f;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int u = row-(K-1)/2+i;
            int v = col-(K-1)/2+j;
            
            if (u >= 0 && u < n && v >= 0 && v < m) res += a[u*m+v]*F_c[i*K+j];
        }
    }
    
    if (row < n && col < m) c[row*m+col] = res;
}

__global__ 
void conv2D_shared_mem(float *a, float *c, int n, int m) {
    __shared__ float a_s[OUT_TILE_WIDTH*OUT_TILE_WIDTH];
    
    // Load the input tile into shared memory
    int row = blockIdx.y*OUT_TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x*OUT_TILE_WIDTH + threadIdx.x;

    if (row < n && col < m) a_s[threadIdx.y*OUT_TILE_WIDTH + threadIdx.x] = a[row*m + col];
    else a_s[threadIdx.y*OUT_TILE_WIDTH + threadIdx.x] = 0.0f;

    __syncthreads();
    
    float res = 0.0f;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int u = threadIdx.y-(K-1)/2+i;
            int v = threadIdx.x-(K-1)/2+j;

            int w = row-(K-1)/2+i;
            int z = col-(K-1)/2+j;

            if (u >= 0 && u < OUT_TILE_WIDTH && v >= 0 && v < OUT_TILE_WIDTH) res += a_s[u*OUT_TILE_WIDTH+v]*F_c[i*K+j];
            else if (w >= 0 && w < n && z >= 0 && z < m) res += a[w*m+z]*F_c[i*K+j];
        }
    }
    
    if (row < n && col < m) c[row*m+col] = res;
}

int main(){
    int n = 1e4;
    int m = 1e3;
    int k = 11;

    float *a, *f, *c1, *c2, *c3, *c4;

    cudaMallocManaged(&a, n*m*sizeof(float));
    cudaMallocManaged(&c1, n*m*sizeof(float));
    cudaMallocManaged(&c2, n*m*sizeof(float));
    cudaMallocManaged(&c3, n*m*sizeof(float));
    cudaMallocManaged(&c4, n*m*sizeof(float));
    cudaMallocManaged(&f, k*k*sizeof(float));

    generate_data(a, n, m);
    generate_data(f, k, k);

    cudaMemcpyToSymbol(F_c, f, K*K*sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    conv2d(a, f, c1, k, n, m);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    // print_vector(c1, 5000, 5100);

    start = std::chrono::high_resolution_clock::now();

    dim3 bd(32, 32, 1);
    dim3 gd(ceil(m/32.0), ceil(n/32.0), 1);

    conv2D_basic<<<gd, bd>>>(a, f, c2, k, n, m);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c2, c1, 0, n*m) << std::endl;

    // print_vector(c2, 5000, 5100);

    start = std::chrono::high_resolution_clock::now();

    conv2D_constant_mem<<<gd, bd>>>(a, c3, n, m);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c3, c1, 0, n*m) << std::endl;

    // print_vector(c3, 5000, 5100);

    start = std::chrono::high_resolution_clock::now();

    dim3 bd1(OUT_TILE_WIDTH, OUT_TILE_WIDTH, 1);
    dim3 gd1(ceil(m/float(OUT_TILE_WIDTH)), ceil(n/float(OUT_TILE_WIDTH)), 1);

    conv2D_shared_mem<<<gd1, bd1>>>(a, c4, n, m);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;
    std::cout << are_equal(c4, c1, 0, n*m) << std::endl;

    // print_vector(c4, 5000, 5100);

}

// 5, 3     0 1 2 3 4  -> 0, 0 1, 0 1 2, 1 2 3, 2 3 4, 3 4, 4