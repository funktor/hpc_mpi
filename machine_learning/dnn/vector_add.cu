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

#define N 50000000
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
    int n = (N-offset)/stride;

    float *a, *out1, *out2, *out3;

    size_t u_size = sizeof(float)*n;

    cudaMallocManaged(&a, u_size);
    cudaMallocManaged(&out1, u_size);
    cudaMallocManaged(&out2, u_size);
    cudaMallocManaged(&out3, u_size);

    generate_data(a, n);

    auto start = std::chrono::high_resolution_clock::now();
    
    normal_add<<<ceil(n/1024.0),1024>>>(a, out1, n);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;



    start = std::chrono::high_resolution_clock::now();
    
    offset_add<<<ceil(n/1024.0),1024>>>(a, out2, n, offset);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;



    start = std::chrono::high_resolution_clock::now();
    
    strided_add<<<ceil(n/1024.0),1024>>>(a, out2, n, stride);
    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;


    cudaFree(a);
    cudaFree(out1);
    cudaFree(out2);
    cudaFree(out3);
}
