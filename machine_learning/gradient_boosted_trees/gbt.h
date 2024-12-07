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

typedef std::pair<double, int> mypair;
// bool comparator ( const mypair& l, const mypair& r) { 
//     return l.first < r.first; 
// }

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
        int sample_features;
        int num_features_to_sample_per_tree;
        std::string model_path;

        GradientBoostedTrees();
        GradientBoostedTrees( 
                int n_features, 
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                int sample_features,
                int num_features_to_sample_per_tree,
                std::string model_path);

        ~GradientBoostedTrees();

        void fit(double *x, double *y, int n);
        double *predict(double *x, int n);
        NodeSplit get_node_split(TreeNode *node, double *scores, double *g, double *h, double *x, double *y, int n);
        NodeSplit get_node_split_feature(TreeNode *node, int feature_index, double *g, double *h, double g_sum, double h_sum, double curr_node_val, double *x, double *y, int n);
};

#endif