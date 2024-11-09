#ifndef SVM_H
#define SVM_H

// #include <mpi.h> 
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
        int d_poly;
        double gamma_rbf;
        double *support_vectors_x;
        int *support_vectors_y;
        double *q_matrix;
        int n_support_vectors;
        std::string model_path;
        std::string kernel;
        double *grad;
        int *up_ind;
        int *lo_ind;

        svm();
        svm(
            int n_features,
            int max_iter, 
            double C, 
            int d_poly,
            double gamma_rbf,
            std::string model_path, 
            std::string kernel);

        ~svm();
        // void distribute_data(double *x, int *y, int n);
        void fit(double *x, int *y, int n);
        // void fit_root(double *x, int *y, int n);
        // void fit_non_root(int n);
        int *predict(double *x, int n);
        // int *predict_root(double *x, int n);
        // void predict_non_root(int n);
        double *predict_proba(double *x, int n);
        void initialize_alpha(int n);
        void initialize_q_matrix(double *x, int *y, int n);
        int update_alpha(double *x, int *y, int n);
        double loss(double *x, int *y, int n);
};

void generate(double *x, int *y, int n, int m);
void save_model_alpha(svm &v, std::string path);
void load_model_alpha(svm &v, std::string path);
double dot_product_vectors(double *a, double *b, int n);
double *dot_product_matrices(double *a, double *b, int n, int m, int p);
double sum_vector(double *a, int n);
// void build_model(double *x, int *y, int n, int n_features, int max_iter, double C, std::string model_path, std::string kernel);
// int *predict_model(double *x, int n, int n_features, std::string model_path);

#endif