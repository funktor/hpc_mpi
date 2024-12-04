#include "gbt.h"
using namespace std;

template <typename T>
void print_arr(const T *arr, const int n, const int m) {
    std::cout << "[";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << arr[i*m+j] << ",";
        }
    }
    std::cout << "]";

    std::cout << std::endl;
    std::cout << std::endl;
}

typedef std::pair<double, int> mypair;
bool comparator ( const mypair& l, const mypair& r) { 
    return l.first < r.first; 
}

GradientBoostedTrees::GradientBoostedTrees(){}

GradientBoostedTrees::GradientBoostedTrees(
                int n_features, 
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                int sample_features,
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

void GradientBoostedTrees::fit(double *x, double *y, int n) {
    int num_tree = 0;
    double *scores = new double [n];
    double *g = new double[n];
    double *h = new double[n];

    for (int i = 0; i < n; i++) scores[i] = 0.0;

    while (num_tree < max_num_trees) {
        std::deque<TreeNode*> nodes;
        
        int *all_indices = new int[n];
        for (int i = 0; i < n; i++) all_indices[i] = i;

        TreeNode *root_node = (TreeNode*) malloc(sizeof(TreeNode));
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
            bool is_leaf = true;

            if (m >= min_samples_for_split && node->depth < max_depth_per_tree) {
                int *curr_indices = node->indices;

                double g_sum = 0.0;
                double h_sum = 0.0;

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];

                    g[j] = -2.0*(y[j]-scores[j]);
                    h[j] = 2.0;

                    g_sum += g[j];
                    h_sum += h[j];
                }

                double curr_node_val = g_sum*g_sum/(h_sum+reg_const);

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
                    is_leaf = false;

                    node->split_feature_index = best_split_feature_index;
                    node->split_feature_value = best_split_feature_value;
                    node->is_leaf = false;

                    TreeNode *lt = (TreeNode*) malloc(sizeof(TreeNode));
                    TreeNode *rt = (TreeNode*) malloc(sizeof(TreeNode));
                    int *lt_indices = new int[best_split_data_index+1];
                    int *rt_indices = new int[m-(best_split_data_index+1)];

                    mypair *features = new mypair[m];
                    for (int j = 0; j < m; j++) {
                        int k = curr_indices[j];
                        features[j] = std::make_pair(x[k*n_features+best_split_feature_index], k);
                    }
                    std::sort(features, features+m, comparator);

                    int p = 0;
                    int q = 0;
                    for (int j = 0; j < m; j++) {
                        mypair z = features[j];
                        int k = z.second;
                        
                        if (j <= best_split_data_index) {
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

            if (is_leaf) {
                node->is_leaf = true;
                int *curr_indices = node->indices;

                double g_sum = 0.0;
                double h_sum = 0.0;

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];

                    g[j] = -2.0*(y[j]-scores[j]);
                    h[j] = 2.0;

                    loss += (y[j]-scores[j])*(y[j]-scores[j]);
                    g_sum += g[j];
                    h_sum += h[j];
                }

                node->leaf_weight = -g_sum/(h_sum+reg_const);

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];
                    new_scores[j] = node->leaf_weight;
                }
            }
        }

        all_trees.push_back(root_node);
        std::cout << "Current Loss = " << loss << std::endl;

        for (int i = 0; i < n; i++) scores[i] += new_scores[i];
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
                    double v = node->split_feature_value;

                    if (x[i*n_features+j] <= v) node = node->lt_node;
                    else node = node->rt_node;
                }
            }
        }
        res[i] = score;
    }

    return res;
}

void generate(double *x, double *y, int n, int m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_n(0.0, 1.0);
    std::uniform_real_distribution<double> dist_u(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            x[i*m+j] = dist_n(engine);
        }

        y[i] = dist_u(engine);
    }
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int n_features = atoi(argv[2]);
    int max_num_trees = atoi(argv[3]);
    int max_depth_per_tree = atoi(argv[4]);
    int min_samples_for_split = atoi(argv[5]);
    double reg_const = atof(argv[6]);
    double gamma = atof(argv[7]);
    std::string model_path = argv[8];

    double *x = new double[n*n_features];
    double *y = new double[n];
    double *y_copy = new double[n];
    generate(x, y, n, n_features);

    std::copy(y, y+n, y_copy);

    GradientBoostedTrees gbt(n_features, max_num_trees, max_depth_per_tree, min_samples_for_split, reg_const, gamma, 0, -1, model_path);
    gbt.fit(x, y, n);
    double *res = gbt.predict(x, n);
    print_arr(res, n, 1);
    std::cout << std::endl;
    print_arr(y_copy, n, 1);

    return 0;
}