#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <mpi.h> 
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

void generate(double *inp, int n, int m);
double *transpose(const double *a, const int n, const int m);
void dot_mpi(const int n, const int m, const int p, const int rank, const int size);
double *dot_mpi_root(const double *a, const double *b, const int n, const int m, const int p, const int size);
double *dot(const double *a, const double *b, const int n, const int m, const int p);

#endif