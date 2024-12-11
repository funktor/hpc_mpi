#include "gbt.h"
using namespace std;

bool custom_comparator ( const mypair& l, const mypair& r) { 
    return l.first < r.first; 
}

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
                double lr,
                double feature_sample,
                double data_sample,
                std::string split_selection_algorithm,
                std::string model_path) {

    this->n_features = n_features;
    this->max_num_trees = max_num_trees;
    this->max_depth_per_tree = max_depth_per_tree;
    this->min_samples_for_split = min_samples_for_split;
    this->reg_const = reg_const;
    this->gamma = gamma;
    this->lr = lr;
    this->feature_sample = feature_sample;
    this->data_sample = data_sample;
    this->split_selection_algorithm = split_selection_algorithm;
    this->model_path = model_path;
    this->bias = 0.0;
    this->curr_feature_importances = new double[n_features];
    this->grad = nullptr;
    this->hess = nullptr;
}

GradientBoostedTrees::~GradientBoostedTrees(){}

int *GradientBoostedTrees::sample_features() {
    mypair *f = new mypair[n_features];

    for (int j = 0; j < n_features; j++) {
        f[j] = std::make_pair(curr_feature_importances[j], j);
    }

    std::sort(f, f+n_features, custom_comparator);
    int n_samples = feature_sample*n_features;

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int *res = new int[n_samples];
    int k = n_features-1;
    int q = 0;
    for (int j = 0; j < n_samples; j++) {
        double h = dist(engine);
        if (h <= 0.99) res[j] = f[k--].second;
        else res[j] = f[q++].second;
    }
    return res;
}

int *GradientBoostedTrees::sample_data(int *curr_indices, int n) {
    mypair *f = new mypair[n];

    for (int j = 0; j < n; j++) {
        int k = curr_indices[j];
        f[j] = std::make_pair(abs(grad[k]), k);
    }

    std::sort(f, f+n, custom_comparator);
    int n_samples = data_sample*n;

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int *res = new int[n_samples];
    int k = n-1;
    int q = 0;
    for (int j = 0; j < n_samples; j++) {
        double h = dist(engine);
        if (h <= 0.99) res[j] = f[k--].second;
        else res[j] = f[q++].second;
    }

    return res;
}

NodeSplit GradientBoostedTrees::get_node_split_feature(TreeNode *node, int feature_index, double g_sum, double h_sum, double curr_node_val, int *curr_indices, int m, double *x, int n) {
    double max_gain = 0;
    int best_split_feature_index = -1;
    double best_split_feature_value = 0.0;

    double g_lt = 0.0;
    double g_rt = g_sum;
    double h_lt = 0.0;
    double h_rt = h_sum;

    if (split_selection_algorithm == "sorting") {
        mypair *f = new mypair[m];

        for (int j = 0; j < m; j++) {
            int k = curr_indices[j];
            f[j] = std::make_pair(x[k*n_features+feature_index], k);
        }

        std::sort(f, f+m, custom_comparator);

        for (int i = 0; i < m; i++) {
            double v = f[i].first;
            int j = f[i].second;

            g_lt += grad[j];
            h_lt += hess[j];
            g_rt -= grad[j];
            h_rt -= hess[j];

            double gain = lr*(1-0.5*lr)*(g_lt*g_lt/(lr*lr*h_lt+reg_const) + g_rt*g_rt/(lr*lr*h_rt+reg_const))-curr_node_val-gamma;

            if (gain > max_gain) {
                best_split_feature_value = v;
                max_gain = gain;
            }
        }
    }
    
    else if (split_selection_algorithm == "histogram") {
        std::vector<std::vector<mypair>> buckets;
        std::vector<mypair> f;

        for (int j = 0; j < m; j++) {
            int k = curr_indices[j];
            f.push_back(std::make_pair(x[k*n_features+feature_index], k));
        }

        buckets.push_back(f);
        
        int num_buckets = 10;
        int max_bucket_size = 20;
        int iters = 0;

        while(iters < 1) {
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
            iters++;
        }

        std::vector<std::vector<mypair>> merged_buckets;
        int curr_len = 0;
        std::vector<mypair> temp;

        for (int i = 0; i < buckets.size(); i++) {
            if (buckets[i].size() > 0) {
                if (curr_len + buckets[i].size() > max_bucket_size && temp.size() > 0) {
                    merged_buckets.push_back(temp);
                    curr_len = 0;
                    temp.clear();
                }

                temp.insert(temp.end(), buckets[i].begin(), buckets[i].end());
                curr_len += buckets[i].size();
            }
        }

        if (temp.size() > 0) merged_buckets.push_back(temp);

        double *g_buckets = new double[merged_buckets.size()];
        double *h_buckets = new double[merged_buckets.size()];
        double *max_val_buckets = new double[merged_buckets.size()];

        for (int i = 0; i < merged_buckets.size(); i++) {
            g_buckets[i] = 0.0;
            h_buckets[i] = 0.0;
            max_val_buckets[i] = -INFINITY;

            for (auto w : merged_buckets[i]) {
                g_buckets[i] += grad[w.second];
                h_buckets[i] += hess[w.second];
                max_val_buckets[i] = max(max_val_buckets[i], w.first);
            }
        }

        for (int i = 0; i < merged_buckets.size(); i++) {
            g_lt += g_buckets[i];
            h_lt += h_buckets[i];
            g_rt -= g_buckets[i];
            h_rt -= h_buckets[i];

            double gain = lr*(1-0.5*lr)*(g_lt*g_lt/(lr*lr*h_lt+reg_const) + g_rt*g_rt/(lr*lr*h_rt+reg_const))-curr_node_val-gamma;

            if (gain > max_gain) {
                best_split_feature_value = max_val_buckets[i];
                max_gain = gain;
            }
        }
    }

    NodeSplit split;

    split.gain = max_gain;
    split.feature_index = feature_index;
    split.split_value = best_split_feature_value;

    return split;
}

NodeSplit GradientBoostedTrees::get_node_split(TreeNode *node, int *sampled_feature_indices, int f_samples, double *x, int n) {
    int m = node->num_indices;

    double max_gain = 0;
    int best_split_feature_index = -1;
    double best_split_feature_value = 0.0;

    if (m >= min_samples_for_split && node->depth < max_depth_per_tree) {
        int *curr_indices;
        int n_samples = data_sample*m;

        if (n_samples > 10) {
            curr_indices = sample_data(node->indices, m);
            m = n_samples;
        }
        else {
            curr_indices = node->indices;
        }

        double g_sum = 0.0;
        double h_sum = 0.0;

        for (int i = 0; i < m; i++) {
            int j = curr_indices[i];
            g_sum += grad[j];
            h_sum += hess[j];
        }

        double curr_node_val = lr*(1-0.5*lr)*g_sum*g_sum/(lr*lr*h_sum+reg_const);

        for (int i = 0; i < f_samples; i++) {
            int j = sampled_feature_indices[i];
            NodeSplit curr_split = get_node_split_feature(node, j, g_sum, h_sum, curr_node_val, curr_indices, m, x, n);
            curr_feature_importances[j] += curr_split.gain;

            if (curr_split.gain > max_gain) {
                best_split_feature_index = j;
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

    double *scores = new double[n];
    grad = new double[n];
    hess = new double[n];

    for (int i = 0; i < n_features; i++) curr_feature_importances[i] = 1.0;

    double u = 0.0;
    for (int i = 0; i < n; i++) u += y[i];
    bias = u/n;

    for (int i = 0; i < n; i++) scores[i] = bias;

    while (num_tree < max_num_trees) {
        std::deque<TreeNode*> nodes;

        for (int i = 0; i < n; i++) {
            grad[i] = -(y[i]-scores[i]);
            hess[i] = 1.0;
        }

        int *all_indices = new int[n];
        for (int i = 0; i < n; i++) all_indices[i] = i;

        int *sampled_feature_indices = sample_features();
        int f_samples = feature_sample*n_features;

        TreeNode *root_node = (TreeNode*) malloc(sizeof(TreeNode));
        root_node->indices = all_indices;
        root_node->num_indices = n;
        root_node->depth = 0;

        nodes.push_back(root_node);

        double *new_scores = new double[n];
        for (int i = 0; i < n; i++) new_scores[i] = scores[i];

        double loss = 0.0;

        while (nodes.size() > 0) {
            TreeNode *node = nodes.front();
            nodes.pop_front();

            int m = node->num_indices;
            int *curr_indices = node->indices;

            bool is_leaf = true;
            NodeSplit split = get_node_split(node, sampled_feature_indices, f_samples, x, n);

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
                    
                    g_sum += grad[j];
                    h_sum += hess[j];
                }

                node->leaf_weight = -lr*g_sum/(lr*lr*h_sum+reg_const);

                for (int i = 0; i < m; i++) {
                    int j = curr_indices[i];
                    new_scores[j] = lr*(node->leaf_weight);
                    loss += 0.5*(y[j]-scores[j]-new_scores[j])*(y[j]-scores[j]-new_scores[j]) + 0.5*reg_const*(node->leaf_weight)*(node->leaf_weight);
                }

                loss += gamma;
            }
            else {
                node->indices = nullptr;
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
        double score = bias;
        for (TreeNode *node : all_trees) {
            while (1) {
                if (node->is_leaf) {
                    score += lr*(node->leaf_weight);
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
    double lr = atof(argv[8]);
    double feature_sample = atof(argv[9]);
    double data_sample = atof(argv[10]);
    std::string split_selection_algorithm = argv[11];
    std::string model_path = argv[12];

    double *x = new double[n*n_features];
    double *y = new double[n];
    double *y_copy = new double[n];
    generate(x, y, n, n_features);

    std::copy(y, y+n, y_copy);

    GradientBoostedTrees gbt(
        n_features, 
        max_num_trees, 
        max_depth_per_tree, 
        min_samples_for_split, 
        reg_const, 
        gamma, 
        lr, 
        feature_sample, 
        data_sample, 
        split_selection_algorithm,
        model_path);

    gbt.fit(x, y, n);
    double *res = gbt.predict(x, n);
    print_arr(res, n, 1);
    std::cout << std::endl;
    print_arr(y_copy, n, 1);

    return 0;
}