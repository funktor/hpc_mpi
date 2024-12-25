#ifndef GBT2_H
#define GBT2_H

#include <mpi.h>
#include <immintrin.h>
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
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

typedef std::pair<double, int> mypair;

struct TreeNode {
    int split_feature_index = -1;
    double split_feature_value = __DBL_MAX__;
    bool is_leaf = false;
    int *indices;
    int num_indices = -1;
    int depth = -1;
    double *leaf_weights;
    TreeNode *lt_node = nullptr;
    TreeNode *rt_node = nullptr;
};

struct NodeSplit {
    double gain;
    int feature_index;
    double split_value;
};

class GradientBoostedTreesClassifier {
    private:

    public:
        std::vector<TreeNode*> all_trees;
        int n_features;
        int n_classes;
        int max_num_trees;
        int max_depth_per_tree;
        int min_samples_for_split;
        double reg_const;
        double gamma;
        double lr;
        double feature_sample;
        double data_sample;
        std::string split_selection_algorithm;
        std::string model_path;
        double bias;
        double *curr_feature_importances;
        double *grad;
        double *hess;
        int rank;
        int n_process;
        int g;
        int start;
        int end;
        MPI_Comm comm;

        GradientBoostedTreesClassifier();
        GradientBoostedTreesClassifier( 
                int n_features, 
                int n_classes,
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                double lr,
                double feature_sample,
                double data_sample,
                std::string split_selection_algorithm,
                std::string model_path,
                MPI_Comm comm);

        ~GradientBoostedTreesClassifier();

        void fit(double *x, int *y, int n);
        int *predict(const double *x, const int n);
        NodeSplit get_node_split(const TreeNode *node, const int *sampled_feature_indices, const int f_samples, double *x, const int n);
        NodeSplit get_node_split_feature(const int feature_index, const double *g_sum, const double *h_sum, const double curr_node_val, const int *curr_indices, const int m, double *x, const int n);
        int *sample_features();
        int *sample_data(const int *curr_indices, const int n);
};

void save_model(GradientBoostedTreesClassifier &gbt, std::string model_path);
GradientBoostedTreesClassifier load_model(std::string model_path);

#endif