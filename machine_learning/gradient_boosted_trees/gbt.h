#ifndef GBT_H
#define GBT_H

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

using namespace std;

typedef std::pair<double, int> mypair;

struct TreeNode {
    int split_feature_index;
    double split_feature_value;
    bool is_leaf;
    int *indices;
    int num_indices;
    int depth;
    double leaf_weight;
    TreeNode *lt_node;
    TreeNode *rt_node;
};

struct NodeSplit {
    double gain;
    int feature_index;
    double split_value;
};

class GradientBoostedTrees {
    private:

    public:
        std::vector<TreeNode*> all_trees;
        int n_features;
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

        GradientBoostedTrees();
        GradientBoostedTrees( 
                int n_features, 
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                double lr,
                double feature_sample,
                double data_sample,
                std::string split_selection_algorithm,
                std::string model_path);

        ~GradientBoostedTrees();

        void fit(double *x, double *y, int n);
        double *predict(double *x, int n);
        NodeSplit get_node_split(TreeNode *node, int *sampled_feature_indices, int f_samples, double *x, int n);
        NodeSplit get_node_split_feature(TreeNode *node, int feature_index, double g_sum, double h_sum, double curr_node_val, int *curr_indices, int m, double *x, int n);
        int *sample_features();
        int *sample_data(int *curr_indices, int n);
};

#endif