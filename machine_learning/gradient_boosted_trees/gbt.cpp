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

NodeSplit GradientBoostedTrees::get_node_split_feature(TreeNode *node, int feature_index, double *g, double *h, double g_sum, double h_sum, double curr_node_val, double *x, double *y, int n) {
    int m = node->num_indices;
    int *curr_indices = node->indices;

    std::vector<std::vector<mypair>> buckets;
    std::vector<mypair> f;

    for (int j = 0; j < m; j++) {
        int k = curr_indices[j];
        f.push_back(std::make_pair(x[k*n_features+feature_index], k));
    }

    buckets.push_back(f);
    
    int num_buckets = 10;
    int max_bucket_size = 20;

    while(1) {
        std::vector<std::vector<mypair>> new_buckets;
        bool flag = false;

        for (auto z : buckets) {
            if (z.size() > max_bucket_size) {
                double a = INFINITY;
                double b = -INFINITY;
                for (auto w : z) {
                    a = min(a, w.first);
                    b = max(b, w.first);
                }

                if (b > a) {
                    flag = true;
                    double interval = (b-a)/(double)num_buckets;

                    std::vector<std::vector<mypair>> f(num_buckets);
                    for (auto w : z) {
                        int p = (w.first-a)/interval;
                        if (p == num_buckets) {
                            f[p-1].push_back(w);
                        }
                        else {
                            f[p].push_back(w);
                        }
                    }
                    new_buckets.insert(new_buckets.end(), f.begin(), f.end());
                }
                else {
                    new_buckets.push_back(z);
                }
                
            }
            else {
                new_buckets.push_back(z);
            }
        }

        buckets.clear();
        buckets.assign(new_buckets.begin(), new_buckets.end());
        if (!flag) break;
    }

    int curr_len = 0;
    std::vector<std::vector<mypair>> merged_buckets;
    std::vector<mypair> temp;

    for (int i = 0; i < buckets.size(); i++) {
        if (curr_len + buckets[i].size() > max_bucket_size && temp.size() > 0) {
            merged_buckets.push_back(temp);
            curr_len = 0;
            temp.clear();
        }

        temp.insert(temp.end(), buckets[i].begin(), buckets[i].end());
        curr_len += buckets[i].size();
    }

    if (temp.size() > 0) merged_buckets.push_back(temp);

    double *g_buckets = new double[merged_buckets.size()];
    double *h_buckets = new double[merged_buckets.size()];

    for (int i = 0; i < merged_buckets.size(); i++) {
        g_buckets[i] = 0.0;
        h_buckets[i] = 0.0;

        for (auto w : merged_buckets[i]) {
            g_buckets[i] += g[w.second];
            h_buckets[i] += h[w.second];
        }
    }

    double max_gain = 0;
    int best_split_feature_index = -1;
    double best_split_feature_value = 0.0;

    double g_lt = 0.0;
    double g_rt = g_sum;
    double h_lt = 0.0;
    double h_rt = h_sum;

    for (int i = 0; i < merged_buckets.size(); i++) {
        g_lt += g_buckets[i];
        h_lt += h_buckets[i];
        g_rt -= g_buckets[i];
        h_rt -= h_buckets[i];

        double gain = 0.5*(g_lt*g_lt/(h_lt+reg_const) + g_rt*g_rt/(h_rt+reg_const) - curr_node_val)-gamma;

        if (gain > max_gain) {
            best_split_feature_value = merged_buckets[i].back().first;
            max_gain = gain;
        }
    }

    NodeSplit split;

    split.gain = max_gain;
    split.feature_index = feature_index;
    split.split_value = best_split_feature_value;

    return split;
}

NodeSplit GradientBoostedTrees::get_node_split(TreeNode *node, double *scores, double *g, double *h, double *x, double *y, int n) {
    int m = node->num_indices;

    double max_gain = 0;
    int best_split_feature_index = -1;
    double best_split_feature_value = 0.0;

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

        for (int i = 0; i < n_features; i++) {
            NodeSplit curr_split = get_node_split_feature(node, i, g, h, g_sum, h_sum, curr_node_val, x, y, n);

            if (curr_split.gain > max_gain) {
                best_split_feature_index = i;
                best_split_feature_value = curr_split.split_value;
                max_gain = curr_split.gain;
            }
        }
    }

    NodeSplit split;

    split.gain = max_gain;
    split.feature_index = best_split_feature_index;
    split.split_value = best_split_feature_value;

    return split;
}

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
            int *curr_indices = node->indices;

            bool is_leaf = true;
            NodeSplit split = get_node_split(node, scores, g, h, x, y, n);
            
            if (split.gain > 0) {
                is_leaf = false;

                node->split_feature_index = split.feature_index;
                node->split_feature_value = split.split_value;
                node->is_leaf = false;

                TreeNode *lt = (TreeNode*) malloc(sizeof(TreeNode));
                TreeNode *rt = (TreeNode*) malloc(sizeof(TreeNode));
                
                int num_lt = 0;
                int num_rt = 0;

                for (int j = 0; j < m; j++) {
                    int k = curr_indices[j];
                    if (x[k*n_features+split.feature_index] <= split.split_value) {
                        num_lt += 1;
                    }
                    else {
                        num_rt += 1;
                    }
                }

                int *lt_indices = new int[num_lt];
                int *rt_indices = new int[num_rt];

                int p = 0;
                int q = 0;

                for (int j = 0; j < m; j++) {
                    int k = curr_indices[j];
                    if (x[k*n_features+split.feature_index] <= split.split_value) {
                        lt_indices[p++] = k;
                    }
                    else {
                        rt_indices[q++] = k;
                    }
                }

                lt->indices = lt_indices;
                rt->indices = rt_indices;

                lt->num_indices = num_lt;
                rt->num_indices = num_rt;

                lt->depth = node->depth+1;
                rt->depth = node->depth+1;

                node->lt_node = lt;
                node->rt_node = rt;

                nodes.push_back(lt);
                nodes.push_back(rt);
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