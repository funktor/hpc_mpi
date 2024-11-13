#ifndef SVM_H
#define SVM_H

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
#include <fstream>
#include <cmath>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

using namespace std;

class svm {
    private:

    public:
        double *alpha;
        double bias;
        int max_iter;
        int n_features;
        double C;
        double *support_vectors_xT;
        int *support_vectors_y;
        double *q_matrix;
        int n_support_vectors;
        std::string model_path;
        double *grad;
        int *up_ind;
        int *lo_ind;
        int rank;
        int n_process;
        MPI_Comm comm;

        svm();
        svm(
            int n_features,
            int max_iter, 
            double C, 
            std::string model_path, 
            MPI_Comm comm);

        ~svm();
        void fit(double *x, int *y, int n);
        int *predict(double *x, int n);
        double *predict_proba(double *x, int n);
        void initialize_alpha(int n);
        int update_alpha(double *x, int *y, int n);
        double loss(double *x, int *y, int n);
};

void generate(double *x, int *y, int n, int m);
void save_model_alpha(svm &v, std::string path);
void load_model_alpha(svm &v, std::string path);
double dot_product_vectors(double *a, double *b, int n);
double *dot_product_matrices(double *a, double *b, int n, int m, int p);
double sum_vector(double *a, int n);

#endif