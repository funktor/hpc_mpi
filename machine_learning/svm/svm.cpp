#include "svm.h"
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

void generate(double *x, int *y, int n, int m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_n(0.0, 1.0);
    std::uniform_int_distribution<int> dist_u(0, 1);

    for (int i = 0; i < n; i++) {
        if (y != nullptr) {
            int z = dist_u(engine);
            y[i] = (z == 0)?-1:1;
        }
        for (int j = 0; j < m; j++) {
            x[i*m+j] = dist_n(engine);
        }
    }
}

void save_model_alpha(svm &v, std::string path) {
    std::ofstream outfile(path);
    if (outfile.is_open()) {
        for (int i = 0; i < v.n_support_vectors*v.n_features; i++) outfile << v.support_vectors_x[i] << " ";
        for (int i = 0; i < v.n_support_vectors; i++) outfile << v.support_vectors_y[i] << " ";
        for (int i = 0; i < v.n_support_vectors; i++) outfile << v.alpha[i] << " ";
        outfile << v.bias;
    }
    outfile.close();
}

void load_model_alpha(svm &v, std::string path) {
    v.alpha = new double[v.n_support_vectors];
    v.support_vectors_x = new double[v.n_support_vectors*v.n_features];
    v.support_vectors_y = new int[v.n_support_vectors];

    std::ifstream infile(path);
    if (infile.is_open()) {
        for (int i = 0; i < v.n_support_vectors*v.n_features; i++) infile >> v.support_vectors_x[i];
        for (int i = 0; i < v.n_support_vectors; i++) infile >> v.support_vectors_y[i];
        for (int i = 0; i < v.n_support_vectors; i++) infile >> v.alpha[i];
        infile >> v.bias;
    }
    infile.close();
}

double dot_product_vectors(double *a, double *b, int n) {
    double out = 0.0;

    for (int j = 0; j < n; j+=8) {
        if (j+8 > n) {
            for (int k = j; k < n; k++) {
                out += a[k]*b[k];
            }
        }
        else {
            __m512d x = _mm512_loadu_pd(&a[j]);
            __m512d y = _mm512_loadu_pd(&b[j]);
            __m512d z = _mm512_mul_pd(x, y);
            out += _mm512_reduce_add_pd(z);
        }
    }

    return out;
}

double *dot_product_matrices(double *a, double *b, int n, int m, int p) {
    double *out = new double[n*p];
    for (int i = 0; i < n*p; i++) out[i] = 0.0;

    if (p > 100) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                __m512d c = _mm512_set1_pd(a[i*m+j]);
                for (int k = 0; k < p; k+=8) {
                    if (k+8 > p) {
                        for (int h = k; h < p; h++) {
                            out[i*p+h] += a[i*m+j]*b[j*p+h];
                        }
                    }
                    else {
                        __m512d x = _mm512_loadu_pd(&b[j*p+k]);
                        __m512d y = _mm512_loadu_pd(&out[i*p+k]);
                        x = _mm512_mul_pd(x, c);
                        y = _mm512_add_pd(y, x);
                        _mm512_storeu_pd(&out[i*p+k], y);
                    }
                }
            }
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < p; k++) {
                    out[i*p+k] += a[i*m+j]*b[j*p+k];
                }
            }
        }
    }

    return out;
}

double sum_vector(double *a, int n) {
    double sum = 0.0;

    for (int i = 0; i < n; i+=8) {
        if (i+8 > n) {
            for (int j = i; j < n; j++) sum += a[j];
        }
        else {
            __m512d x = _mm512_loadu_pd(&a[i]);
            sum += _mm512_reduce_add_pd(x);
        }
    }

    return sum;
}

svm::svm(){}

svm::svm(
        int n_features,
        int max_iter, 
        double C, 
        int d_poly,
        double gamma_rbf,
        std::string model_path, 
        std::string kernel) {

    this->n_features = n_features;
    this->max_iter = max_iter;
    this->C = C;
    this->d_poly = d_poly;
    this->gamma_rbf = gamma_rbf;
    this->model_path = model_path;
    this->kernel = kernel;
}

svm::~svm(){}

void svm::initialize_alpha(int n) {
    alpha = new double[n];
    for (int i = 0; i < n; i++) alpha[i] = 0.0;
    bias = 0.0;
}

void svm::initialize_q_matrix(double *x, int *y, int n) {
    q_matrix = new double[n*n];

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            q_matrix[i*n+j] = y[i]*y[j]*dot_product_vectors(&x[i*n_features], &x[j*n_features], n_features);
            q_matrix[j*n+i] = q_matrix[i*n+j];
        }
    }
}

double *svm::predict_proba(double *x, int n) {
    double *out = new double[n];

    for (int i = 0; i < n; i++) {
        out[i] = bias;
        for (int j = 0; j < n_support_vectors; j++) {
            out[i] += support_vectors_y[j]*alpha[j]*dot_product_vectors(&support_vectors_x[j*n_features], &x[i*n_features], n_features);
        }
    }

    return out;
}

int *svm::predict(double *x, int n) {
    load_model_alpha(*this, model_path);
    double *scores = predict_proba(x, n);

    int *res = new int[n];
    for (int i = 0; i < n; i++) {
        if (scores[i] >= 0.0) res[i] = 1;
        else res[i] = -1;
    }

    return res;
}

double svm::loss(double *x, int *y, int n) {
    double *a = dot_product_matrices(alpha, q_matrix, 1, n, n);
    double b = dot_product_vectors(a, alpha, n);
    double c = sum_vector(alpha, n);
    return 0.5*b-c;
}

int svm::update_alpha(double *x, int *y, int n) {
    int ind_0 = -1;
    double max_ind_0 = -INFINITY;

    for (int i = 0; i < n; i++) {
        if (up_ind[i] == 1) {
            double u = -y[i]*grad[i];
            if (u > max_ind_0) {
                max_ind_0 = u;
                ind_0 = i;
            }
        }
    }

    if (ind_0 == -1) return -1;

    int ind_1 = -1;
    double min_ind_1 = INFINITY;

    for (int i = 0; i < n; i++) {
        if (lo_ind[i] == 1) {
            double u = -y[i]*grad[i];
            if (u < max_ind_0) {
                double p = q_matrix[ind_0*n+ind_0]+q_matrix[i*n+i]-2.0*q_matrix[ind_0*n+i]/(y[ind_0]*y[i]);
                p = (p >= 0.0)?p:1e-10;
                double q = max_ind_0-u;
                double r = -q*q/p;
                if (r < min_ind_1) {
                    min_ind_1 = r;
                    ind_1 = i;
                }
            }
        }
    } 

    if (ind_1 != -1 && max_ind_0-min_ind_1 > 1e-3) {
        double l, h;

        if (y[ind_0] != y[ind_1]) {
            l = max(0.0, alpha[ind_1]-alpha[ind_0]);
            h = min(C, C+alpha[ind_1]-alpha[ind_0]);
        }
        else {
            l = max(0.0, alpha[ind_1]+alpha[ind_0]-C);
            h = min(C, alpha[ind_1]+alpha[ind_0]);
        }

        int s = y[ind_0]*y[ind_1];

        double p = q_matrix[ind_0*n+ind_0]+q_matrix[ind_1*n+ind_1]-2.0*q_matrix[ind_0*n+ind_1]/s;
        p = (p >= 0.0)?p:1e-10;
        double q = -y[ind_0]*grad[ind_0]+y[ind_1]*grad[ind_1];
        double old_alpha_0 = alpha[ind_0];
        double old_alpha_1 = alpha[ind_1];
        alpha[ind_1] -= y[ind_1]*q/p;

        if (alpha[ind_1] <= l) alpha[ind_1] = l;
        else if (alpha[ind_1] >= h) alpha[ind_1] = h;

        alpha[ind_0] += s*(old_alpha_1-alpha[ind_1]);

        double diff_0 = alpha[ind_0]-old_alpha_0;
        double diff_1 = alpha[ind_1]-old_alpha_1;

        for (int i = 0; i < n; i++) {
            grad[i] += q_matrix[i*n+ind_0]*diff_0 + q_matrix[i*n+ind_1]*diff_1;
        }

        up_ind[ind_0] = ((alpha[ind_0] < C && y[ind_0] == 1) || (alpha[ind_0] > 0 && y[ind_0] == -1))?1:0;
        up_ind[ind_1] = ((alpha[ind_1] < C && y[ind_1] == 1) || (alpha[ind_1] > 0 && y[ind_1] == -1))?1:0;
        lo_ind[ind_0] = ((alpha[ind_0] < C && y[ind_0] == -1) || (alpha[ind_0] > 0 && y[ind_0] == 1))?1:0;
        lo_ind[ind_1] = ((alpha[ind_1] < C && y[ind_1] == -1) || (alpha[ind_1] > 0 && y[ind_1] == 1))?1:0;

        return 1;
    }

    return -1;
}

void svm::fit(double *x, int *y, int n) {
    n_support_vectors = 0;
    up_ind = new int[n];
    lo_ind = new int[n];

    initialize_alpha(n);
    initialize_q_matrix(x, y, n);

    for (int i = 0; i < n; i++) {
        up_ind[i] = ((alpha[i] < C && y[i] == 1) || (alpha[i] > 0 && y[i] == -1))?1:0;
        lo_ind[i] = ((alpha[i] < C && y[i] == -1) || (alpha[i] > 0 && y[i] == 1))?1:0;
    }

    grad = dot_product_matrices(q_matrix, alpha, n, n, 1);
    for (int i = 0; i < n; i++) grad[i] -= 1.0;

    int n_iter = max_iter;
    while (n_iter > 0) {
        int r = update_alpha(x, y, n);
        double l = loss(x, y, n);
        std::cout << "Current Loss = " << l << std::endl;
        if (r == -1) break;
        n_iter--;
    }

    double h1 = 0.0;
    int h2 = 0;

    for (int i = 0; i < n; i++) {
        if (alpha[i] > 0 && alpha[i] < C) {
            h2 += 1;
            h1 += y[i]*grad[i];
        }
    }

    bias = -h1/h2; 
    
    for (int i = 0; i < n; i++) {
        if (alpha[i] > 1e-10) n_support_vectors++;
    }

    double *old_alpha = new double[n];
    std::copy(alpha, alpha+n, old_alpha);

    support_vectors_x = new double[n_support_vectors*n_features];
    support_vectors_y = new int[n_support_vectors];
    alpha = new double[n_support_vectors];

    int j = 0;
    for (int i = 0; i < n; i++) {
        if (old_alpha[i] > 1e-10) {
            std::copy(&x[i*n_features], &x[(i+1)*n_features], &support_vectors_x[j*n_features]);
            support_vectors_y[j] = y[i];
            alpha[j] = old_alpha[i];
            j++;
        }
    }

    save_model_alpha(*this, model_path);
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int n_features = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    double C = atof(argv[4]);
    int d_poly = atof(argv[5]);
    double gamma_rbf = atof(argv[6]);
    std::string model_path = argv[7];
    std::string kernel = argv[8];

    double *x = new double[n*n_features];
    int *y = new int[n];
    generate(x, y, n, n_features);

    svm v(n_features, max_iter, C, d_poly, gamma_rbf, model_path, kernel);
    v.fit(x, y, n);

    int *res = v.predict(x, n);

    int h = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] == res[i]) h++;
    }

    std::cout << h/(double)n << std::endl;
    
    return 0;
}