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
        for (int i = 0; i < v.n_support_vectors*v.n_features; i++) outfile << v.support_vectors_xT[i] << " ";
        for (int i = 0; i < v.n_support_vectors; i++) outfile << v.support_vectors_y[i] << " ";
        for (int i = 0; i < v.n_support_vectors; i++) outfile << v.alpha[i] << " ";
        outfile << v.bias;
    }
    outfile.close();
}

void load_model_alpha(svm &v, std::string path) {
    v.alpha = new double[v.n_support_vectors];
    v.support_vectors_xT = new double[v.n_support_vectors*v.n_features];
    v.support_vectors_y = new int[v.n_support_vectors];

    std::ifstream infile(path);
    if (infile.is_open()) {
        for (int i = 0; i < v.n_support_vectors*v.n_features; i++) infile >> v.support_vectors_xT[i];
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
        std::string kernel, 
        MPI_Comm comm) {

    MPI_Comm_size(comm, &this->n_process);
    MPI_Comm_rank(comm, &this->rank);

    this->n_features = n_features;
    this->max_iter = max_iter;
    this->C = C;
    this->d_poly = d_poly;
    this->gamma_rbf = gamma_rbf;
    this->model_path = model_path;
    this->kernel = kernel;
    this->comm = comm;
}

svm::~svm(){}

void svm::initialize_alpha(int n) {
    alpha = new double[n];
    for (int i = 0; i < n; i++) alpha[i] = 0.0;
    bias = 0.0;
}

void svm::initialize_q_matrix(double *x, int *y, int n) {
    double *xT = new double[n*n_features];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_features; j++) {
            xT[j*n+i] = x[i*n_features+j];
        }
    }

    q_matrix = dot_product_matrices(x, xT, n, n_features, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            q_matrix[i*n+j] *= y[i]*y[j];
        }
    }
}

void svm::distribute_data(double *x, int *y, int n) {
    MPI_Request request = MPI_REQUEST_NULL;

    int h = n/n_process;
    int m = n % n_process;

    for (int p = 1; p < n_process; p++) {
        if (p+1 <= m) {
            int u = h+1;
            MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
            if (y != nullptr) MPI_Isend(y+p*u, u, MPI_INT, p, p, comm, &request);
        }
        else {
            if (p <= m) {
                int u = h+1;
                MPI_Isend(x+p*u*n_features, h*n_features, MPI_DOUBLE, p, p, comm, &request);
                if (y != nullptr) MPI_Isend(y+p*u, h, MPI_INT, p, p, comm, &request);
            }
            else {
                int u = h;
                MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
                if (y != nullptr) MPI_Isend(y+p*u, u, MPI_INT, p, p, comm, &request);
            }
        }
    }
}

double *svm::predict_proba(double *x, int n) {
    double *out = new double[n];
    for (int i = 0; i < n; i++) out[i] = 0.0;

    if (n_support_vectors > 0) {
        double *u = dot_product_matrices(x, support_vectors_xT, n, n_features, n_support_vectors);
        double *v = new double[n_support_vectors];
        for (int i = 0; i < n_support_vectors; i++) v[i] = support_vectors_y[i]*alpha[i];
        out = dot_product_matrices(u, v, n, n_support_vectors, 1);
    }
    
    return out;
}

int *svm::predict(double *x, int n) {
    load_model_alpha(*this, model_path + "_" + std::to_string(rank));
    double *scores = new double[n];
    int *res = new int[n];

    if (rank == 0) {
        scores = predict_proba(x, n);

        MPI_Send(x, n*n_features, MPI_DOUBLE, 1, 0, comm);
        MPI_Send(scores, n, MPI_DOUBLE, 1, 1, comm);

        MPI_Recv(x, n*n_features, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(scores, n, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);
        
        for (int i = 0; i < n; i++) {
            if (scores[i]+bias >= 0.0) res[i] = 1;
            else res[i] = -1;
        }
    }
    else {
        double *scores1 = new double[n];
        x = new double[n*n_features];

        MPI_Recv(x, n*n_features, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(scores1, n, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
        scores = predict_proba(x, n);

        for (int i = 0; i < n; i++) scores[i] += scores1[i];

        MPI_Send(x, n*n_features, MPI_DOUBLE, (rank+1)%n_process, 0, comm);
        MPI_Send(scores, n, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
    }

    return res;
}

double svm::loss(double *x, int *y, int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g, start, end;

    if (rank == 0) {
        g = (m == 0)?h:h+1;
        start = 0;
        end = g-1;
    }
    else {
        g = (rank+1 <= m)?h+1:h;
        start = (rank <= m)?rank*(h+1):rank*h;
        end = start+g-1;
    }

    double *a = dot_product_matrices(alpha+start, q_matrix, 1, g, n);
    double b = dot_product_vectors(a, alpha, n);
    double c = sum_vector(alpha+start, g);
    return 0.5*b-c;
}

int svm::update_alpha(double *x, int *y, int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g, start, end;

    if (rank == 0) {
        g = (m == 0)?h:h+1;
        start = 0;
        end = g-1;
    }
    else {
        g = (rank+1 <= m)?h+1:h;
        start = (rank <= m)?rank*(h+1):rank*h;
        end = start+g-1;
    }

    int ind_0 = -1;
    double max_ind_0 = -INFINITY;
    double alpha_0 = INFINITY;
    double grad_0 = INFINITY;
    int y_0 = 0;
    double *q_matrix_0 = new double[n];

    for (int i = 0; i < g; i++) {
        if (up_ind[i] == 1) {
            double u = -y[i]*grad[i];
            if (u > max_ind_0) {
                max_ind_0 = u;
                ind_0 = i;
                alpha_0 = alpha[i+start];
                grad_0 = grad[i];
                y_0 = y[i];
            }
        }
    }

    double *data = new double[n+5];
    std::copy(q_matrix+ind_0*n, q_matrix+(ind_0+1)*n, data);
    data[n] = (ind_0 != -1)?(double)ind_0+start:-1.0;
    data[n+1] = max_ind_0;
    data[n+2] = alpha_0;
    data[n+3] = grad_0;
    data[n+4] = (double) y_0;

    if (rank == 0) {
        MPI_Send(data, n+5, MPI_DOUBLE, 1, 0, comm);
        MPI_Recv(data, n+5, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);

        MPI_Send(data, n+5, MPI_DOUBLE, 1, 1, comm);
        MPI_Recv(data, n+5, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);
    }
    else {
        double *recv_data = new double[n+5];
        MPI_Recv(recv_data, n+5, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);

        if (recv_data[n+1] > max_ind_0) std::copy(recv_data, recv_data+n+5, data);

        MPI_Send(data, n+5, MPI_DOUBLE, (rank+1)%n_process, 0, comm);
        MPI_Recv(data, n+5, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
        MPI_Send(data, n+5, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
    }

    ind_0 = (int) data[n];
    max_ind_0 = data[n+1];
    alpha_0 = data[n+2];
    grad_0 = data[n+3];
    y_0 = (int) data[n+4];
    std::copy(data, data+n, q_matrix_0);

    int ind_1 = -1;
    double min_ind_1 = INFINITY;
    double alpha_1 = INFINITY;
    double grad_1 = INFINITY;
    int y_1 = 0;
    double r_min = INFINITY;
    double *q_matrix_1 = new double[n];

    if (ind_0 != -1) {
        for (int i = 0; i < g; i++) {
            if (lo_ind[i] == 1) {
                double u = -y[i]*grad[i];
                min_ind_1 = min(min_ind_1, u);

                if (u < max_ind_0) {
                    double p = q_matrix_0[ind_0]+q_matrix[i*n+i+start]-2.0*q_matrix_0[i+start]/(y_0*y[i]);
                    p = (p >= 0.0)?p:1e-10;
                    double q = max_ind_0-u;
                    double r = -q*q/p;
                    if (r < r_min) {
                        r_min = r;
                        ind_1 = i;
                        alpha_1 = alpha[i+start];
                        grad_1 = grad[i];
                        y_1 = y[i];
                    }
                }
            }
        }
    }

    data = new double[n+6];
    std::copy(q_matrix+ind_1*n, q_matrix+(ind_1+1)*n, data);
    data[n] = (ind_1 != -1)?(double)ind_1+start:-1.0;
    data[n+1] = min_ind_1;
    data[n+2] = alpha_1;
    data[n+3] = grad_1;
    data[n+4] = (double) y_1;
    data[n+5] = r_min;

    if (rank == 0) {
        MPI_Send(data, n+6, MPI_DOUBLE, 1, 2, comm);
        MPI_Recv(data, n+6, MPI_DOUBLE, n_process-1, 2, comm, MPI_STATUS_IGNORE);
        MPI_Send(data, n+6, MPI_DOUBLE, 1, 3, comm);
        MPI_Recv(data, n+6, MPI_DOUBLE, n_process-1, 3, comm, MPI_STATUS_IGNORE);
    }
    else {
        double *recv_data = new double[n+6];
        MPI_Recv(recv_data, n+6, MPI_DOUBLE, (rank-1)%n_process, 2, comm, MPI_STATUS_IGNORE);

        if (recv_data[n+5] < r_min) std::copy(recv_data, recv_data+n+6, data);

        MPI_Send(data, n+6, MPI_DOUBLE, (rank+1)%n_process, 2, comm);
        MPI_Recv(data, n+6, MPI_DOUBLE, (rank-1)%n_process, 3, comm, MPI_STATUS_IGNORE);
        MPI_Send(data, n+6, MPI_DOUBLE, (rank+1)%n_process, 3, comm);
    }

    ind_1 = (int) data[n];
    min_ind_1 = data[n+1];
    alpha_1 = data[n+2];
    grad_1 = data[n+3];
    y_1 = (int) data[n+4];
    std::copy(data, data+n, q_matrix_1);

    std::cout << rank << " " << ind_0 << " " << ind_1 << " " << max_ind_0-min_ind_1 << " " << std::endl;

    if (ind_1 != -1 && max_ind_0-min_ind_1 > 1e-3) {
        double l, h;

        if (y_0 != y_1) {
            l = max(0.0, alpha_1-alpha_0);
            h = min(C, C+alpha_1-alpha_0);
        }
        else {
            l = max(0.0, alpha_1+alpha_0-C);
            h = min(C, alpha_1+alpha_0);
        }

        int s = y_0*y_1;

        double p = q_matrix_0[ind_0]+q_matrix_1[ind_1]-2.0*q_matrix_0[ind_1]/s;
        p = (p >= 0.0)?p:1e-10;
        double q = -y_0*grad_0+y_1*grad_1;
        double old_alpha_0 = alpha_0;
        double old_alpha_1 = alpha_1;
        alpha_1 -= y_1*q/p;

        if (alpha_1 <= l) alpha_1 = l;
        else if (alpha_1 >= h) alpha_1 = h;

        alpha_0 += s*(old_alpha_1-alpha_1);

        double diff_0 = alpha_0-old_alpha_0;
        double diff_1 = alpha_1-old_alpha_1;

        for (int i = 0; i < g; i++) {
            grad[i] += q_matrix[i*n+ind_0]*diff_0 + q_matrix[i*n+ind_1]*diff_1;
        }

        alpha[ind_0] = alpha_0;
        alpha[ind_1] = alpha_1;

        if (ind_0 >= start && ind_0 <= end) {
            up_ind[ind_0-start] = ((alpha[ind_0] < C && y[ind_0-start] == 1) || (alpha[ind_0] > 0 && y[ind_0-start] == -1))?1:0;
            lo_ind[ind_0-start] = ((alpha[ind_0] < C && y[ind_0-start] == -1) || (alpha[ind_0] > 0 && y[ind_0-start] == 1))?1:0;
        }

        if (ind_1 >= start && ind_1 <= end) {
            up_ind[ind_1-start] = ((alpha[ind_1] < C && y[ind_1-start] == 1) || (alpha[ind_1] > 0 && y[ind_1-start] == -1))?1:0;
            lo_ind[ind_1-start] = ((alpha[ind_1] < C && y[ind_1-start] == -1) || (alpha[ind_1] > 0 && y[ind_1-start] == 1))?1:0;
        }

        return 1;
    }

    return -1;
}

void svm::fit(double *x, int *y, int n) {
    MPI_Request request = MPI_REQUEST_NULL;

    int h = n/n_process;
    int m = n % n_process;
    int g, start, end;

    if (rank == 0) {
        double *xT = new double[n*n_features];
                
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n_features; j++) {
                xT[j*n+i] = x[i*n_features+j];
            }
        }

        g = (m == 0)?h:h+1;
        start = 0;
        end = g-1;

        q_matrix = dot_product_matrices(x, xT, g, n_features, n);

        for (int i = 0; i < g; i++) {
            for (int j = 0; j < n; j++) {
                q_matrix[i*n+j] *= y[i]*y[j];
            }
        }

        for (int p = 1; p < n_process; p++) {
            if (p+1 <= m) {
                int u = h+1;

                double *q_matrix_p = new double[u*n];
                q_matrix_p = dot_product_matrices(x+p*u*n_features, xT, u, n_features, n);

                for (int i = p*u; i < (p+1)*u; i++) {
                    for (int j = 0; j < n; j++) {
                        q_matrix_p[(i-p*u)*n+j] *= y[i]*y[j];
                    }
                }

                MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
                MPI_Isend(q_matrix_p, u*n, MPI_DOUBLE, p, p+1, comm, &request);
                MPI_Isend(y+p*u, u, MPI_INT, p, p+2, comm, &request);
            }

            else {
                if (p <= m) {
                    int u = h+1;

                    double *q_matrix_p = dot_product_matrices(x+p*u*n_features, xT, h, n_features, n);

                    for (int i = p*u; i < p*u+h; i++) {
                        for (int j = 0; j < n; j++) {
                            q_matrix_p[(i-p*u)*n+j] *= y[i]*y[j];
                        }
                    }

                    MPI_Isend(x+p*u*n_features, h*n_features, MPI_DOUBLE, p, p, comm, &request);
                    MPI_Isend(q_matrix_p, h*n, MPI_DOUBLE, p, p+1, comm, &request);
                    MPI_Isend(y+p*u, h, MPI_INT, p, p+2, comm, &request);
                }
                else {
                    int u = h;

                    double *q_matrix_p = new double[u*n];
                    q_matrix_p = dot_product_matrices(x+p*u*n_features, xT, u, n_features, n);

                    for (int i = p*u; i < p*u+u; i++) {
                        for (int j = 0; j < n; j++) {
                            q_matrix_p[(i-p*u)*n+j] *= y[i]*y[j];
                        }
                    }

                    MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
                    MPI_Isend(q_matrix_p, u*n, MPI_DOUBLE, p, p+1, comm, &request);
                    MPI_Isend(y+p*u, u, MPI_INT, p, p+2, comm, &request);
                }
            }
        }

    }
    else {
        g = (rank+1 <= m)?h+1:h;
        start = (rank <= m)?rank*(h+1):rank*h;
        end = start+g-1;

        x = new double[g*n_features];
        q_matrix = new double[g*n];
        y = new int[g];

        MPI_Recv(x, g*n_features, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
        MPI_Recv(q_matrix, g*n, MPI_DOUBLE, 0, rank+1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(y, g, MPI_INT, 0, rank+2, comm, MPI_STATUS_IGNORE);
    }

    n_support_vectors = 0;

    up_ind = new int[g];
    lo_ind = new int[g];

    initialize_alpha(n);

    for (int i = 0; i < g; i++) {
        up_ind[i] = ((alpha[i+start] < C && y[i] == 1) || (alpha[i+start] > 0 && y[i] == -1))?1:0;
        lo_ind[i] = ((alpha[i+start] < C && y[i] == -1) || (alpha[i+start] > 0 && y[i] == 1))?1:0;
    }

    grad = dot_product_matrices(q_matrix, alpha, g, n, 1);
    for (int i = 0; i < g; i++) grad[i] -= 1.0;

    int n_iter = max_iter;
    while (n_iter > 0) {
        int r = update_alpha(x, y, n);
        double l = loss(x, y, n);

        if (rank == 0) {
            MPI_Send(&l, 1, MPI_DOUBLE, 1, 0, comm);
            MPI_Recv(&l, 1, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);
            MPI_Send(&l, 1, MPI_DOUBLE, 1, 1, comm);
            MPI_Recv(&l, 1, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);
            std::cout << "Current Loss = " << l << std::endl;
        }
        else {
            double l1 = 0.0;
            MPI_Recv(&l1, 1, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);
            
            double l2 = l+l1;
            MPI_Send(&l2, 1, MPI_DOUBLE, (rank+1)%n_process, 0, comm);
            MPI_Recv(&l, 1, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
            MPI_Send(&l, 1, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
        }

        if (r == -1) break;
        n_iter--;
    }

    double h1 = 0.0;
    int h2 = 0;

    for (int i = 0; i < g; i++) {
        if (alpha[i+start] > 0 && alpha[i+start] < C) {
            h2 += 1;
            h1 += y[i]*grad[i];
        }
    }

    double *hh = new double[2];
    hh[0] = h1;
    hh[1] = (double) h2;

    if (rank == 0) {
        MPI_Send(hh, 2, MPI_DOUBLE, 1, 0, comm);
        MPI_Recv(hh, 2, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(hh, 2, MPI_DOUBLE, 1, 1, comm);
        MPI_Recv(hh, 2, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);
    }
    else {
        double *hh1 = new double[2];
        MPI_Recv(hh1, 2, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);
        hh[0] += hh1[0];
        hh[1] += hh1[1]; 
        MPI_Send(hh, 2, MPI_DOUBLE, (rank+1)%n_process, 0, comm);
        MPI_Recv(hh, 2, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
        MPI_Send(hh, 2, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
    }

    bias = -hh[0]/hh[1]; 

    std::cout << rank << " " << bias << std::endl;
    print_arr(alpha, 1, n);
    
    for (int i = 0; i < g; i++) {
        if (alpha[i+start] > 1e-10) n_support_vectors++;
    }

    double *old_alpha = new double[g];
    std::copy(alpha+start, alpha+end+1, old_alpha);

    double *support_vectors = new double[n_support_vectors*n_features];
    
    support_vectors_xT = new double[n_support_vectors*n_features];
    support_vectors_y = new int[n_support_vectors];
    alpha = new double[n_support_vectors];

    int j = 0;
    for (int i = 0; i < g; i++) {
        if (old_alpha[i] > 1e-10) {
            std::copy(&x[i*n_features], &x[(i+1)*n_features], &support_vectors[j*n_features]);
            support_vectors_y[j] = y[i];
            alpha[j] = old_alpha[i];
            j++;
        }
    }

    for (int i = 0; i < n_support_vectors; i++) {
        for (int j = 0; j < n_features; j++) {
            support_vectors_xT[j*n_support_vectors+i] = support_vectors[i*n_features+j];
        }
    }

    save_model_alpha(*this, model_path + "_" + std::to_string(rank));
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

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *x = new double[n*n_features];
    int *y = new int[n];
    generate(x, y, n, n_features);

    svm v(n_features, max_iter, C, d_poly, gamma_rbf, model_path, kernel, MPI_COMM_WORLD);
    v.fit(x, y, n);

    int *res = v.predict(x, n);

    int h = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] == res[i]) h++;
    }

    std::cout << h/(double)n << std::endl;
    
    MPI_Finalize();
    return 0;
}