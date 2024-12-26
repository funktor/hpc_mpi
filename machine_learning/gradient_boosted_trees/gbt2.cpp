#include "gbt2.h"
using namespace std;

double *transpose(const double *a, const int n, const int m) {
    double *b = new double[n*m];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[j*n+i] = a[i*m+j];
        }
    }

    return b;
}

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

double *least_squares(double *x, double *y, int n, int m, double lr, int num_epochs) {
    double *w = new double[m];
    double *preds = new double[n];
    double *grads = new double[m];

    for (int i = 0; i < m; i++) w[i] = 0;

    while (num_epochs > 0) {
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < m; j++) {
                s += w[j]*x[i*m+j];
            }
            preds[i] = s;
        }
        
        for (int j = 0; j < m; j++) grads[j] = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                grads[j] += 1.0/n*(preds[i]-y[i])*x[i*m+j];
            }
        }

        for (int j = 0; j < m; j++) w[j] -= lr*grads[j];
        num_epochs--;
    }

    delete[] preds;
    delete[] grads;

    return w;
}

GradientBoostedTreesClassifier::GradientBoostedTreesClassifier(){}

GradientBoostedTreesClassifier::GradientBoostedTreesClassifier(
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
                MPI_Comm comm) {

    MPI_Comm_size(comm, &this->n_process);
    MPI_Comm_rank(comm, &this->rank);

    this->n_features = n_features;
    this->n_classes = n_classes;
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
    this->comm = comm;
    this->g = 0;
    this->start = 0;
    this->end = 0;
}

GradientBoostedTreesClassifier::~GradientBoostedTreesClassifier(){}

int *GradientBoostedTreesClassifier::sample_features() {
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

    delete[] f;

    return res;
}

int *GradientBoostedTreesClassifier::sample_data(const int *curr_indices, const int n) {
    mypair *f = new mypair[n];

    for (int j = 0; j < n; j++) {
        int k = curr_indices[j];
        double gg = 0.0;
        for (int cl = 0; cl < n_classes; cl++) {
            gg += grad[k*n_classes+cl]*grad[k*n_classes+cl];
        }
        f[j] = std::make_pair(gg, k);
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

    delete[] f;

    return res;
}

NodeSplit GradientBoostedTreesClassifier::get_node_split_feature(const int feature_index, const double *g_sum, const double *h_sum, const double curr_node_val, const int *curr_indices, const int m, double *x, const int n) {
    double max_gain = -INFINITY;
    int best_split_feature_index = -1;
    double best_split_feature_value = __DBL_MAX__;

    double *g_lt = new double[n_classes];
    double *g_rt = new double[n_classes];
    double *h_lt = new double[n_classes];
    double *h_rt = new double[n_classes];

    std::copy(g_sum, g_sum+n_classes, g_rt);
    std::copy(h_sum, h_sum+n_classes, h_rt);

    if (split_selection_algorithm == "sorting") {
        mypair *f = new mypair[m];

        for (int j = 0; j < m; j++) {
            int k = curr_indices[j];
            f[j] = std::make_pair(x[feature_index*n+k], k);
        }

        std::sort(f, f+m, custom_comparator);

        double *gs = new double[n_classes];
        double *hs = new double[n_classes];

        for (int cl = 0; cl < n_classes; cl++) {
            gs[cl] = 0.0;
            hs[cl] = 0.0;
        }

        for (int i = 0; i < m; i++) {
            double v = f[i].first;
            int j = f[i].second;

            if (i+1 < m && v == f[i+1].first) {
                for (int cl = 0; cl < n_classes; cl++) {
                    gs[cl] += grad[j*n_classes+cl];
                    hs[cl] += hess[j*n_classes+cl];
                }
            }
            else {
                for (int cl = 0; cl < n_classes; cl++) {
                    gs[cl] += grad[j*n_classes+cl];
                    hs[cl] += hess[j*n_classes+cl];
                }

                for (int cl = 0; cl < n_classes; cl++) {
                    g_lt[cl] += gs[cl];
                    h_lt[cl] += hs[cl];
                    g_rt[cl] -= gs[cl];
                    h_rt[cl] -= hs[cl];
                }

                double gg1 = 0.0;
                double gg2 = 0.0;
                for (int cl = 0; cl < n_classes; cl++) {
                    gg1 += 0.5*lr*lr*g_lt[cl]*g_lt[cl]/(lr*lr*h_lt[cl]+reg_const);
                    gg2 += 0.5*lr*lr*g_rt[cl]*g_rt[cl]/(lr*lr*h_rt[cl]+reg_const);
                }

                double gain = gg1+gg2-curr_node_val-gamma;

                if (gain > max_gain) {
                    best_split_feature_value = v;
                    max_gain = gain;
                }
                
                for (int cl = 0; cl < n_classes; cl++) {
                    gs[cl] = 0.0;
                    hs[cl] = 0.0;
                }
            }
        }

        delete[] f;
    }
    
    else if (split_selection_algorithm == "histogram") {
        std::vector<std::vector<mypair>> buckets;
        std::vector<mypair> f;

        for (int j = 0; j < m; j++) {
            int k = curr_indices[j];
            f.push_back(std::make_pair(x[feature_index*n+k], k));
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

        double *g_buckets = new double[n_classes*merged_buckets.size()];
        double *h_buckets = new double[n_classes*merged_buckets.size()];
        double *max_val_buckets = new double[merged_buckets.size()];

        for (int i = 0; i < merged_buckets.size(); i++) {
            for (int cl = 0; cl < n_classes; cl++) {
                g_buckets[i*n_classes+cl] = 0.0;
                h_buckets[i*n_classes+cl] = 0.0;
            }
            max_val_buckets[i] = -INFINITY;

            for (auto w : merged_buckets[i]) {
                for (int cl = 0; cl < n_classes; cl++) {
                    g_buckets[i*n_classes+cl] += grad[w.second*n_classes+cl];
                    h_buckets[i*n_classes+cl] += hess[w.second*n_classes+cl];
                }
                max_val_buckets[i] = max(max_val_buckets[i], w.first);
            }
        }

        for (int i = 0; i < merged_buckets.size(); i++) {
            for (int cl = 0; cl < n_classes; cl++) {
                g_lt[cl] += g_buckets[i*n_classes+cl];
                h_lt[cl] += h_buckets[i*n_classes+cl];
                g_rt[cl] -= g_buckets[i*n_classes+cl];
                h_rt[cl] -= h_buckets[i*n_classes+cl];
            }
            
            double gg1 = 0.0;
            double gg2 = 0.0;
            for (int cl = 0; cl < n_classes; cl++) {
                gg1 += 0.5*lr*lr*g_lt[cl]*g_lt[cl]/(lr*lr*h_lt[cl]+reg_const);
                gg2 += 0.5*lr*lr*g_rt[cl]*g_rt[cl]/(lr*lr*h_rt[cl]+reg_const);
            }

            double gain = gg1+gg2-curr_node_val-gamma;

            if (gain > max_gain) {
                best_split_feature_value = max_val_buckets[i];
                max_gain = gain;
            }
        }

        delete[] g_buckets;
        delete[] h_buckets;
        delete[] max_val_buckets;
    }

    NodeSplit split;

    split.gain = max_gain;
    split.feature_index = feature_index;
    split.split_value = best_split_feature_value;

    return split;
}

NodeSplit GradientBoostedTreesClassifier::get_node_split(const TreeNode *node, const int *sampled_feature_indices, const int f_samples, double *x, const int n) {
    int m = node->num_indices;

    double max_gain = -INFINITY;
    int best_split_feature_index = -1;
    double best_split_feature_value = __DBL_MAX__;

    if (m >= min_samples_for_split && node->depth < max_depth_per_tree) {
        int *curr_indices = new int[m];
        int n_samples = data_sample*m;

        if (n_samples > 10) {
            if (rank == 0) {
                curr_indices = sample_data(node->indices, m);
                m = n_samples;

                for (int p = 1; p < n_process; p++) {
                    MPI_Send(curr_indices, n_samples, MPI_INT, p, p, comm);
                }
            }
            else {
                MPI_Recv(curr_indices, n_samples, MPI_INT, 0, rank, comm, MPI_STATUS_IGNORE);
                m = n_samples;
            }
        }
        else {
            std::copy(node->indices, node->indices+m, curr_indices);
        }

        double *g_sum = new double[n_classes];
        double *h_sum = new double[n_classes];

        for (int cl = 0; cl < n_classes; cl++) {
            g_sum[cl] = 0.0;
            h_sum[cl] = 0.0;
        }

        for (int i = 0; i < m; i++) {
            int j = curr_indices[i];

            for (int cl = 0; cl < n_classes; cl++) {
                g_sum[cl] += grad[j*n_classes+cl];
                h_sum[cl] += hess[j*n_classes+cl];
            }
        }

        double curr_node_val = 0.0;
        for (int cl = 0; cl < n_classes; cl++) {
            curr_node_val += 0.5*lr*lr*g_sum[cl]*g_sum[cl]/(lr*lr*h_sum[cl]+reg_const);
        }

        for (int i = 0; i < f_samples; i++) {
            int j = sampled_feature_indices[i];
            if (j >= start && j <= end) {
                NodeSplit curr_split = get_node_split_feature(j-start, g_sum, h_sum, curr_node_val, curr_indices, m, x, n);
                curr_feature_importances[j] += curr_split.gain;

                if (curr_split.gain > max_gain) {
                    best_split_feature_index = j;
                    best_split_feature_value = curr_split.split_value;
                    max_gain = curr_split.gain;
                }
            }
        }

        delete[] curr_indices;
    }

    NodeSplit split;

    split.gain = max_gain;
    split.feature_index = best_split_feature_index;
    split.split_value = best_split_feature_value;

    return split;
}

void GradientBoostedTreesClassifier::fit(double *x, int *y, const int n) {
    if (rank == 0) x = transpose(x, n, n_features);

    MPI_Request request = MPI_REQUEST_NULL;

    int h = n_features/n_process;
    int m = n_features % n_process;

    if (rank == 0) {
        g = (m == 0)?h:h+1;
        start = 0;
        end = g-1;
        
        for (int p = 1; p < n_process; p++) {
            if (p+1 <= m) {
                int u = h+1;
                MPI_Isend(x+p*u*n, u*n, MPI_DOUBLE, p, p, comm, &request);
            }

            else {
                if (p <= m) {
                    int u = h+1;
                    MPI_Isend(x+p*u*n, h*n, MPI_DOUBLE, p, p, comm, &request);
                }
                else {
                    int u = h;
                    MPI_Isend(x+(p*u+m)*n, u*n, MPI_DOUBLE, p, p, comm, &request);
                }
            }

            MPI_Isend(y, n, MPI_INT, p, p+1, comm, &request);
        }
    }
    else {
        g = (rank+1 <= m)?h+1:h;
        start = (rank <= m)?(rank*(h+1)):(m+rank*h);
        end = start+g-1;

        x = new double[g*n];
        y = new int[n];

        MPI_Recv(x, g*n, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
        MPI_Recv(y, n, MPI_INT, 0, rank+1, comm, MPI_STATUS_IGNORE);
    }
    
    int num_tree = 0;

    double *scores = new double[n*n_classes];
    grad = new double[n*n_classes];
    hess = new double[n*n_classes];
    int *labels = new int[n*n_classes];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_classes; j++) labels[i*n_classes+j] = 0;
        labels[i*n_classes+y[i]] = 1;
    }

    for (int i = 0; i < n_features; i++) curr_feature_importances[i] = 1.0/n_features;

    bias = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_classes; j++) {
            scores[i*n_classes+j] = bias;
        }
    }

    while (num_tree < max_num_trees) {
        std::deque<TreeNode*> nodes;

        for (int i = 0; i < n; i++) {
            double maxs = -INFINITY;
            double mins = INFINITY;
            for (int cl = 0; cl < n_classes; cl++) {
                maxs = max(maxs, scores[i*n_classes+cl]);
                mins = min(mins, scores[i*n_classes+cl]);
            }

            double ss1 = 0.0;
            double ss2 = 0.0;
            for (int j = 0; j < n_classes; j++) {
                ss1 += exp(scores[i*n_classes+j]-maxs);
                ss2 += exp(scores[i*n_classes+j]-mins);
            }

            for (int j = 0; j < n_classes; j++) {
                double z;
                if (ss1 == 0) z = (ss2 == 0)?0.5:exp(scores[i*n_classes+j]-mins)/ss2;
                else z = exp(scores[i*n_classes+j]-maxs)/ss1;

                grad[i*n_classes+j] = z-labels[i*n_classes+j];
                hess[i*n_classes+j] = z*(1-z);
            }
        }

        int f_samples = feature_sample*n_features;
        int *sampled_feature_indices = new int[f_samples];

        if (rank == 0) {
            sampled_feature_indices = sample_features();

            for (int p = 1; p < n_process; p++) {
                MPI_Send(sampled_feature_indices, f_samples, MPI_INT, p, p, comm);
            }
        }
        else {
            MPI_Recv(sampled_feature_indices, f_samples, MPI_INT, 0, rank, comm, MPI_STATUS_IGNORE);
        }

        TreeNode *root_node = new TreeNode;
        root_node->indices = new int[n];
        for (int i = 0; i < n; i++) root_node->indices[i] = i;
        root_node->num_indices = n;
        root_node->depth = 0;

        nodes.push_back(root_node);

        double *new_scores = new double[n*n_classes];
        std::copy(scores, scores+n*n_classes, new_scores);

        double loss = 0.0;

        while (nodes.size() > 0) {
            TreeNode *node = nodes.front();
            nodes.pop_front();

            int m = node->num_indices;
            int *curr_indices = node->indices;

            node->leaf_weights = new double[n_classes];
            for (int cl = 0; cl < n_classes; cl++) node->leaf_weights[cl] = 0.0;

            bool is_leaf = true;
            NodeSplit split = get_node_split(node, sampled_feature_indices, f_samples, x, n);
            
            int *lt_rt_indices = new int[m];

            if (split.gain > 1e-10 && split.feature_index >= start && split.feature_index <= end) {
                for (int j = 0; j < m; j++) {
                    int k = curr_indices[j];
                    if (x[(split.feature_index-start)*n+k] <= split.split_value) {
                        lt_rt_indices[j] = 0;
                    }
                    else {
                        lt_rt_indices[j] = 1;
                    }
                }
            } 
            else {
                for (int j = 0; j < m; j++) {
                    lt_rt_indices[j] = 0;
                }
            }

            double *split_data = new double[m+n_features+3];
            double *split_data_recv = new double[m+n_features+3];
            
            split_data[0] = split.feature_index;
            split_data[1] = split.split_value;
            split_data[2] = split.gain;
            std::copy(lt_rt_indices, lt_rt_indices+m, split_data+3);
            std::copy(curr_feature_importances, curr_feature_importances+n_features, split_data+m+3);

            if (rank == 0) {
                MPI_Send(split_data, m+n_features+3, MPI_DOUBLE, 1, 0, comm);
                MPI_Recv(split_data_recv, m+n_features+3, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);

                if (split_data_recv[2] > split_data[2]) std::copy(split_data_recv, split_data_recv+m+3, split_data);
                std::copy(split_data_recv+m+3, split_data_recv+m+n_features+3, split_data+m+3);

                MPI_Send(split_data, m+n_features+3, MPI_DOUBLE, 1, 1, comm);
                MPI_Recv(split_data_recv, m+n_features+3, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);
                
                std::copy(split_data_recv, split_data_recv+m+n_features+3, split_data);
            }
            else {
                double *split_data_recv = new double[m+n_features+3];
                MPI_Recv(split_data_recv, m+n_features+3, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);
                if (split_data_recv[2] > split_data[2]) std::copy(split_data_recv, split_data_recv+m+3, split_data);
                
                for (int j = 0; j < n_features; j++) {
                    if (j < start || j > end) {
                        split_data[m+j+3] = split_data_recv[m+j+3];
                    }
                }
                
                MPI_Send(split_data, m+n_features+3, MPI_DOUBLE, (rank+1)%n_process, 0, comm);

                split_data_recv = new double[m+n_features+3];
                MPI_Recv(split_data_recv, m+n_features+3, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
                std::copy(split_data_recv, split_data_recv+m+n_features+3, split_data);

                MPI_Send(split_data, m+n_features+3, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
            }

            delete[] split_data_recv;

            split.feature_index = split_data[0];
            split.split_value = split_data[1];
            split.gain = split_data[2];
            std::copy(split_data+3, split_data+m+3, lt_rt_indices);
            std::copy(split_data+m+3, split_data+m+n_features+3, curr_feature_importances);

            delete[] split_data;

            if (split.gain > 1e-10) {
                int num_lt = 0;
                int num_rt = 0;

                for (int j = 0; j < m; j++) {
                    if (lt_rt_indices[j] == 0) {
                        num_lt++;
                    }
                    else {
                        num_rt++;
                    }
                }

                if (num_rt > 0 && num_lt > 0) {
                    is_leaf = false;

                    node->split_feature_index = split.feature_index;
                    node->split_feature_value = split.split_value;
                    node->is_leaf = false;

                    TreeNode *lt = new TreeNode;
                    TreeNode *rt = new TreeNode;

                    lt->indices = new int[num_lt];
                    rt->indices = new int[num_rt];

                    int p = 0;
                    int q = 0;

                    for (int j = 0; j < m; j++) {
                        int k = curr_indices[j];
                        if (lt_rt_indices[j] == 0) {
                            lt->indices[p++] = k;
                        }
                        else {
                            rt->indices[q++] = k;
                        }
                    }

                    lt->num_indices = num_lt;
                    rt->num_indices = num_rt;

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

                double *g_sum = new double[n_classes];
                double *h_sum = new double[n_classes];

                for (int cl = 0; cl < n_classes; cl++) {
                    g_sum[cl] = 0.0;
                    h_sum[cl] = 0.0;
                }

                for (int i = 0; i < m; i++) {
                    int j = node->indices[i];
                    for (int cl = 0; cl < n_classes; cl++) {
                        g_sum[cl] += grad[j*n_classes+cl];
                        h_sum[cl] += hess[j*n_classes+cl];
                    }
                }

                double mw = -INFINITY;
                for (int cl = 0; cl < n_classes; cl++) {
                    node->leaf_weights[cl] = -lr*g_sum[cl]/(lr*lr*h_sum[cl]+reg_const);
                    mw = max(mw, abs(node->leaf_weights[cl]));
                }

                mw = max(mw, 1e-3);

                for (int cl = 0; cl < n_classes; cl++) node->leaf_weights[cl] /= mw;

                delete[] g_sum;
                delete[] h_sum;

                for (int i = 0; i < m; i++) {
                    int j = node->indices[i];

                    for (int cl = 0; cl < n_classes; cl++) new_scores[j*n_classes+cl] = lr*(node->leaf_weights[cl]);

                    double maxs = -INFINITY;
                    double mins = INFINITY;
                    for (int cl = 0; cl < n_classes; cl++) {
                        maxs = max(maxs, scores[j*n_classes+cl]+new_scores[j*n_classes+cl]);
                        mins = min(mins, scores[j*n_classes+cl]+new_scores[j*n_classes+cl]);
                    }

                    double ss1 = 0.0;
                    double ss2 = 0.0;
                    for (int cl = 0; cl < n_classes; cl++) {
                        ss1 += exp(scores[j*n_classes+cl]+new_scores[j*n_classes+cl]-maxs);
                        ss2 += exp(scores[j*n_classes+cl]+new_scores[j*n_classes+cl]-mins);
                    }

                    for (int cl = 0; cl < n_classes; cl++) {
                        double z;
                        if (ss1 == 0) z = (ss2 == 0)?0.5:exp(scores[j*n_classes+cl]+new_scores[j*n_classes+cl]-mins)/ss2;
                        else z = exp(scores[j*n_classes+cl]+new_scores[j*n_classes+cl]-maxs)/ss1;

                        loss += (z == 0)?0.0:-labels[j*n_classes+cl]*log(z)*(1.0/n);
                    }
                }
            }
            else {
                delete[] node->indices;
            }

            delete[] lt_rt_indices;
        }

        all_trees.push_back(root_node);

        if (rank == 0) {
            std::cout << "Current Loss = " << loss << std::endl;
        }

        for (int i = 0; i < n*n_classes; i++) scores[i] += new_scores[i];
        delete[] new_scores;

        num_tree++;
    }

    delete[] scores;
    delete[] grad;
    delete[] hess;
}

int *GradientBoostedTreesClassifier::predict(const double *x, const int n) {
    int *res = new int[n];

    for (int i = 0; i < n; i++) {
        double *scores = new double[n_classes];
        for (int cl = 0; cl < n_classes; cl++) scores[cl] = bias;

        for (TreeNode *node : all_trees) {
            while (node != nullptr) {
                if (node->is_leaf) {
                    for (int cl = 0; cl < n_classes; cl++) scores[cl] += lr*(node->leaf_weights[cl]);
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

        double maxs = -INFINITY;
        double mins = INFINITY;
        for (int cl = 0; cl < n_classes; cl++) {
            maxs = max(maxs, scores[cl]);
            mins = min(mins, scores[cl]);
        }

        double ss1 = 0;
        double ss2 = 0;
        for (int cl = 0; cl < n_classes; cl++) {
            ss1 += exp(scores[cl]-maxs);
            ss2 += exp(scores[cl]-mins);
        }

        double max_sss = -INFINITY;
        int best_cl = -1;

        for (int cl = 0; cl < n_classes; cl++) {
            double sss;
            if (ss1 == 0) sss = (ss2 == 0)?0.5:exp(scores[cl]-mins)/ss2;
            else sss = exp(scores[cl]-maxs)/ss1;

            if (sss > max_sss) {
                max_sss = sss;
                best_cl = cl;
            }
        }

        res[i] = best_cl;
    }

    return res;
}

struct NodeSer {
    TreeNode *node;
    int parent_id;
    int is_left;
};

void save_model(GradientBoostedTreesClassifier &gbt, const std::string model_path) {
    std::ofstream outfile(model_path);

    if (outfile.is_open()) {
        outfile << gbt.n_features << " ";
        outfile << gbt.n_classes << " ";
        outfile << gbt.lr << " ";
        outfile << gbt.bias << " ";

        int curr_node_id = 0;

        for (TreeNode *root : gbt.all_trees) {
            std::deque<NodeSer> qtrees;
            NodeSer q = {root, -1, -1};
            qtrees.push_back(q);

            while (qtrees.size() > 0) {
                NodeSer qt = qtrees.front();
                qtrees.pop_front();

                TreeNode *node = qt.node;
                outfile << curr_node_id << " ";
                outfile << qt.parent_id << " ";
                outfile << qt.is_left << " ";
                outfile << node->is_leaf << " ";

                for (int cl = 0; cl < gbt.n_classes; cl++) outfile << node->leaf_weights[cl] << " ";
                
                outfile << node->split_feature_index << " ";
                outfile << node->split_feature_value << " ";

                if (node->lt_node != nullptr) {
                    NodeSer q = {node->lt_node, curr_node_id, 1};
                    qtrees.push_back(q);
                }
                if (node->rt_node != nullptr) {
                    NodeSer q = {node->rt_node, curr_node_id, 0};
                    qtrees.push_back(q);
                }

                curr_node_id++;
            }
        }

        outfile << -1 << " ";
    }

    outfile.close();
    std::cout << "Object serialized successfully." << std::endl;
}

GradientBoostedTreesClassifier load_model(const std::string model_path) {
    GradientBoostedTreesClassifier gbt;
    std::ifstream infile(model_path);

    if (infile.is_open()) {
        infile >> gbt.n_features;
        infile >> gbt.n_classes;
        infile >> gbt.lr;
        infile >> gbt.bias;

        std::vector<TreeNode *> node_seq;

        while (1) {
            TreeNode *node = new TreeNode;
            int curr_node_id;
            int parent_node_id;
            int is_left;

            infile >> curr_node_id;
            if (curr_node_id == -1) break;

            infile >> parent_node_id;
            infile >> is_left;
            infile >> node->is_leaf;

            node->leaf_weights = new double[gbt.n_classes];
            for (int cl = 0; cl < gbt.n_classes; cl++) infile >> node->leaf_weights[cl];
            
            infile >> node->split_feature_index;
            infile >> node->split_feature_value;

            if (parent_node_id == -1) gbt.all_trees.push_back(node);
            
            if (is_left == 1) node_seq[parent_node_id]->lt_node = node;
            else if (is_left == 0) node_seq[parent_node_id]->rt_node = node;

            node_seq.push_back(node);
        }
    }

    infile.close();
    return gbt;
}

void generate(double *x, int *y, int n, int m, int n_classes) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_n(0.0, 1.0);
    std::uniform_int_distribution<int> dist_u(0, n_classes-1);

    for (int i = 0; i < n; i++) {
        if (y != nullptr) y[i] = dist_u(engine);
        for (int j = 0; j < m; j++) {
            x[i*m+j] = dist_n(engine);
        }
    }
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int n_features = atoi(argv[2]);
    int n_classes = atoi(argv[3]);
    int max_num_trees = atoi(argv[4]);
    int max_depth_per_tree = atoi(argv[5]);
    int min_samples_for_split = atoi(argv[6]);
    double reg_const = atof(argv[7]);
    double gamma = atof(argv[8]);
    double lr = atof(argv[9]);
    double feature_sample = atof(argv[10]);
    double data_sample = atof(argv[11]);
    std::string split_selection_algorithm = argv[12];
    std::string model_path = argv[13];

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *x;
    int *y;

    if (rank == 0) {
        x = new double[n*n_features];
        y = new int[n];
        generate(x, y, n, n_features, n_classes);
    }

    GradientBoostedTreesClassifier gbt(
        n_features, 
        n_classes,
        max_num_trees, 
        max_depth_per_tree, 
        min_samples_for_split, 
        reg_const, 
        gamma, 
        lr, 
        feature_sample, 
        data_sample, 
        split_selection_algorithm,
        model_path, 
        MPI_COMM_WORLD);

    gbt.fit(x, y, n);

    if (rank == 0) save_model(gbt, gbt.model_path);

    if (rank == 0) {
        GradientBoostedTreesClassifier gbt2 = load_model(gbt.model_path);
        int *res = gbt2.predict(x, n);
        print_arr(res, n, 1);
        std::cout << std::endl;
        print_arr(y, n, 1);
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            if (res[i] == y[i]) s++;
        }
        std::cout << s/n << std::endl;
        delete[] res;
    }

    MPI_Finalize();
    return 0;
}