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

#define N 10000000
#define MAX_ERR 1e-6

void generate_data(float *x, int n) {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < n; i++) x[i] = dist(rng);
}

__global__
void normal_add(float *inp, float *oup, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) oup[index] = inp[index] + 100.0;
}

__global__
void offset_add(float *inp, float *oup, int n, int offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index += offset;
    oup[index] = inp[index] + 100.0;
}

__global__
void strided_add(float *inp, float *oup, int n, int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= stride;
    oup[index] = inp[index] + 100.0;
}


int main(int argc, char **argv){
    int offset = atoi(argv[1]);
    int stride = atoi(argv[2]);
    int n1 = N+offset;
    int n2 = N*stride;

    float *a, *b, *c, *out1, *out2, *out3;

    cudaMallocManaged(&a, sizeof(float)*N);
    cudaMallocManaged(&b, sizeof(float)*n1);
    cudaMallocManaged(&c, sizeof(float)*n2);

    cudaMallocManaged(&out1, sizeof(float)*N);
    cudaMallocManaged(&out2, sizeof(float)*n1);
    cudaMallocManaged(&out3, sizeof(float)*n2);

    generate_data(a, N);
    generate_data(b, n1);
    generate_data(c, n2);

    auto start = std::chrono::high_resolution_clock::now();
    
    normal_add<<<ceil(N/1024.0),1024>>>(a, out1, N);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;



    start = std::chrono::high_resolution_clock::now();
    
    offset_add<<<ceil(N/1024.0),1024>>>(b, out2, n1, offset);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;



    start = std::chrono::high_resolution_clock::now();
    
    strided_add<<<ceil(N/1024.0),1024>>>(c, out3, n2, stride);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(out1);
    cudaFree(out2);
    cudaFree(out3);
}
