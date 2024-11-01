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
        int num_epochs;
        int batch_size;
        int num_features;
        double l1_reg;
        double l2_reg;
        std::string model_save_location;
        logistic_regression();
        logistic_regression(
                double learning_rate, 
                int num_epochs, 
                int batch_size, 
                int num_features, 
                double l1_reg,
                double l2_reg,
                std::string model_save_location);

        ~logistic_regression();

        void fit(
                double *x_train, 
                int *y_train, 
                int n);

        int *predict(double *x_test, int n);
        double *predict_proba(double *x_test, int n);
        void initialize_weights(int n);
        void update_weights_and_biases(
                double *x_train, 
                int *y_train, 
                int n);
        double loss(
                double *x_train, 
                int *y_train, 
                int n);
};

void distribute_data(double *x, int *y, int n, int n_features, int n_process);
void lr_train(int n, int n_features, int rank, int n_process, double learning_rate, int epochs, int batch_size);
void lr_train_root(double *x, int *y, int n, int n_features, int n_process, double learning_rate, int epochs, int batch_size);
void build_model(double *x, int *y, int n, int n_features, double learning_rate, int epochs, int batch_size);

#endif