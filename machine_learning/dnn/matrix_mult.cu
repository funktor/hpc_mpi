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
#define TILE_WIDTH 16

void print_vector(float *x, size_t n) {
    std::cout << "[";
    for (int i = 0; i < n; i++) std::cout << x[i] << ", ";
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

int main(){
    int n = 2048;
    int m = 2048;
    int p = 2048;

    float *a, *b, *c, *d;
    float *d_a, *d_b, *d_c; 

    size_t size_a = sizeof(float)*n*m;
    size_t size_b = sizeof(float)*m*p;
    size_t size_c = sizeof(float)*n*p;

    a   = (float*)malloc(size_a);
    b   = (float*)malloc(size_b);
    c   = (float*)malloc(size_c);
    d   = (float*)malloc(size_c);

    generate_data(a, n, m);
    generate_data(b, m, p);

    auto start = std::chrono::high_resolution_clock::now();
    
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

    dim3 bd(16, 16, 1);
    dim3 gd(ceil(n/16.0), ceil(p/16.0), 1);

    cuda_mul_tiled<<<gd, bd>>>(d_a, d_b, d_c, n, m, p);
    
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mat_mul(a, b, d, n, m, p);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;

    free(a); 
    free(b); 
    free(c);
    free(d);
}
