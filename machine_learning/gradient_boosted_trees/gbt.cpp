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

double *transpose(const double *a, const int *indices, const int n, const int m) {
    double *b = new double[n*m];

    for (int i = 0; i < n; i++) {
        int k = indices[i];
        for (int j = 0; j < m; j++) {
            b[j*n+k] = a[k*m+j];
        }
    }

    return b;
}

struct TreeNode {
    int split_feature_index;
    double split_feature_value;
    bool is_leaf;
    int *indices;
    int num_indices;
    double *scores;
    int depth;
    double leaf_weight;
    TreeNode *lt_node;
    TreeNode *rt_node;
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
        bool sample_features;
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
                bool sample_features,
                int num_features_to_sample_per_tree,
                std::string model_path);

        ~GradientBoostedTrees();

        void fit(double *x, double *y, int n);
        double *predict(double *x, int n);
};

GradientBoostedTrees::GradientBoostedTrees(){}

GradientBoostedTrees::GradientBoostedTrees(
                int n_features, 
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                bool sample_features,
                int num_features_to_sample_per_tree,
                std::string model_path) {

    this->n_features = n_features;
    this->max_num_trees = max_num_trees;
    this->max_depth_per_tree = max_depth_per_tree;
    this->min_samples_for_split = min_samples_for_split;
    this->reg_const = reg_const;
    this->gamma = gamma;
    this->sample_features = sample_features;
    this->num_features_to_sample_per_tree = num_features_to_sample_per_tree;
    this->model_path = model_path;
}

GradientBoostedTrees::~GradientBoostedTrees(){}

typedef std::pair<double, int> mypair;
bool comparator ( const mypair& l, const mypair& r) { 
    return l.first < r.first; 
}

void GradientBoostedTrees::fit(double *x, double *y, int n) {
    int num_tree = 0;
    double *curr_scores = new double [n];
    double *g = new double[n];
    double *h = new double[n];

    for (int i = 0; i < n; i++) curr_scores[i] = 0.0;

    while (num_tree < max_num_trees) {
        std::deque<TreeNode*> nodes;
        TreeNode *root_node;
        int *all_indices = new int[n];
        for (int i = 0; i < n; i++) all_indices[i] = i;
        root_node->indices = all_indices;
        root_node->num_indices = n;
        root_node->depth = 0;
        nodes.push_back(root_node);

        double *new_scores = new double[n];
        double loss = 0.0;

        while (nodes.size() > 0) {
            TreeNode *node = nodes.front();
            nodes.pop_front();

            int m = node->num_indices;

            if (m >= min_samples_for_split && node->depth < max_depth_per_tree) {
                int *curr_indices = node->indices;

                double g_sum = 0.0;
                double h_sum = 0.0;

                double xx = 0.0;
                double yy = 0.0;

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];

                    g[j] = -2.0*(y[j]-curr_scores[j]);
                    h[j] = 2.0;

                    g_sum += g[j];
                    h_sum += h[j];

                    xx += g[j]*g[j];
                    yy += h[j];
                }

                double curr_node_val = xx/(yy+reg_const);

                double max_gain = -INFINITY;
                int best_split_feature_index = -1;
                int best_split_data_index = -1;
                double best_split_feature_value = 0.0;

                for (int i = 0; i < n_features; i++) {
                    mypair *features = new mypair[m];
                    for (int j = 0; j < m; j++) {
                        int k = curr_indices[j];
                        features[j] = std::make_pair(x[k*n_features+i], k);
                    }
                    std::sort(features, features+m, comparator);

                    int *lt_indices = new int[m];
                    int *rt_indices = new int[m];

                    for (int k = 0; k < m; k++) {
                        lt_indices[k] = 0;
                        rt_indices[k] = 1;
                    }

                    double g_lt = 0.0;
                    double g_rt = g_sum;
                    double h_lt = 0.0;
                    double h_rt = h_sum;

                    for (int j = 0; j < m; j++) {
                        mypair p = features[j];
                        int k = p.second;

                        g_lt += g[k];
                        h_lt += h[k];
                        g_rt -= g[k];
                        h_rt -= h[k];

                        lt_indices[k] = 1;
                        rt_indices[k] = 0;

                        double gain = 0.5*(g_lt*g_lt/(h_lt+reg_const) + g_rt*g_rt/(h_rt+reg_const) - curr_node_val)-gamma;
                        
                        if (gain > max_gain) {
                            best_split_feature_index = i;
                            best_split_data_index = j;
                            best_split_feature_value = p.first;
                            max_gain = gain;
                        }
                    }
                }

                if (max_gain > 0) {
                    int i = best_split_feature_index;
                    node->split_feature_index = best_split_feature_index;
                    node->split_feature_value = best_split_feature_value;
                    node->is_leaf = false;

                    TreeNode *lt, *rt;
                    int *lt_indices = new int[best_split_data_index+1];
                    int *rt_indices = new int[m-(best_split_data_index+1)];

                    int p = 0;
                    int q = 0;
                    for (int j = 0; j < m; j++) {
                        int k = curr_indices[j];
                        if (x[k*n_features+best_split_feature_index] <= best_split_feature_value) {
                            lt_indices[p++] = k;
                        }
                        else {
                            rt_indices[q++] = k;
                        }
                    }

                    lt->indices = lt_indices;
                    rt->indices = rt_indices;

                    lt->num_indices = best_split_data_index+1;
                    rt->num_indices = m-(best_split_data_index+1);

                    lt->depth = node->depth+1;
                    rt->depth = node->depth+1;

                    node->lt_node = lt;
                    node->rt_node = rt;

                    nodes.push_back(lt);
                    nodes.push_back(rt);
                }
            }
            else {
                node->is_leaf = true;
                int *curr_indices = node->indices;

                double g_sum = 0.0;
                double h_sum = 0.0;

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];

                    g[j] = -2.0*(y[j]-curr_scores[j]);
                    h[j] = 2.0;

                    g_sum += g[j];
                    h_sum += h[j];
                }

                node->leaf_weight = -g_sum/(h_sum+reg_const);
                loss += -0.5*(g_sum*g_sum)/(h_sum+reg_const) + gamma;

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];
                    new_scores[j] = curr_scores[j] + node->leaf_weight;
                }
            }
        }

        all_trees.push_back(root_node);
        std::cout << "Current Loss = " << loss << std::endl;

        std::copy(new_scores, new_scores+n, curr_scores);
        for (int i = 0; i < n; i++) y[i] -= curr_scores[i];

        num_tree++;
    }
}

double *GradientBoostedTrees::predict(double *x, int n) {
    double *res = new double[n];
    
    for (int i = 0; i < n; i++) {
        double score = 0.0;
        for (TreeNode *node : all_trees) {
            while (1) {
                if (node->is_leaf) {
                    score += node->leaf_weight;
                    break;
                }
                else {
                    int j = node->split_feature_index;
                    double k = node->split_feature_value;

                    if (x[i*n_features+j] <= k) node = node->lt_node;
                    else node = node->rt_node;
                }
            }
        }
        res[i] = score;
    }

    return res;
}