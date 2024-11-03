#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

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

class logistic_regression {
    private:

    public:
        double *weights;
        double bias;
        double learning_rate;
        int epochs;
        int batch_size;
        int n_features;
        double l1_reg;
        double l2_reg;
        std::string model_path;
        int rank;
        int n_process;
        MPI_Comm comm;

        logistic_regression();
        logistic_regression(
                double learning_rate, 
                int epochs, 
                int batch_size, 
                int n_features, 
                double l1_reg,
                double l2_reg, 
                std::string model_path,
                MPI_Comm comm);

        ~logistic_regression();

        void distribute_data(double *x, unsigned int *y, int n);
        void fit_root(double *x, unsigned int *y, int n);
        void fit_non_root(int n);
        unsigned int *predict(double *x, int n);
        unsigned int *predict_root(double *x, int n);
        void predict_non_root(int n);
        double *predict_proba(double *x, int n);
        void initialize_weights(int n);
        void update_weights_and_biases(double *x, unsigned int *y, int n);
        double loss(double *x, unsigned int *y, int n);
};

void generate(double *x, unsigned int *y, int n, int m);
void save_model_weights(logistic_regression &lr, std::string path);
void load_model_weights(logistic_regression &lr, std::string path);
void build_model(double *x, unsigned int *y, int n, int n_features, double learning_rate, int epochs, int batch_size, double l1_reg, double l2_reg, std::string model_path);
unsigned int *predict_model(double *x, int n, int n_features, std::string model_path);

#endif