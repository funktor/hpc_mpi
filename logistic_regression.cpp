#include "logistic_regression.h"
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

void generate(double *x, unsigned int *y, int n, int m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_n(0.0, 1.0);
    std::uniform_int_distribution<unsigned int> dist_u(0, 1);

    for (int i = 0; i < n; i++) {
        if (y != nullptr) y[i] = dist_u(engine);
        for (int j = 0; j < m; j++) {
            x[i*m+j] = dist_n(engine);
        }
    }
}

void save_model_weights(logistic_regression &lr, std::string path) {
    std::ofstream outfile(path);
    if (outfile.is_open()) {
        for (int i = 0; i < lr.n_features; i++) outfile << lr.weights[i] << " ";
        outfile << lr.bias;
    }
    outfile.close();
}

void load_model_weights(logistic_regression &lr, std::string path) {
    lr.weights = new double[lr.n_features];

    std::ifstream infile(path);
    if (infile.is_open()) {
        for (int i = 0; i < lr.n_features; i++) infile >> lr.weights[i];
        infile >> lr.bias;
    }
    infile.close();
}

logistic_regression::logistic_regression(){}

logistic_regression::logistic_regression(
                double learning_rate, 
                int epochs, 
                int batch_size, 
                int n_features, 
                double l1_reg,
                double l2_reg, 
                std::string model_path,
                MPI_Comm comm) {
    
    MPI_Comm_size(comm, &this->n_process);
    MPI_Comm_rank(comm, &this->rank);

    this->learning_rate = learning_rate;
    this->epochs = epochs;
    this->batch_size = batch_size;
    this->n_features = n_features;
    this->l1_reg = l1_reg;
    this->l2_reg = l2_reg;
    this->model_path = model_path;
    this->comm = comm;
}

logistic_regression::~logistic_regression(){}

void logistic_regression::initialize_weights(int n) {
    weights = new double[n_features];

    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0/n);
    for (int i = 0; i < n_features; i++) weights[i] = dist(engine);
    bias = 1.0/n;
}

double *logistic_regression::predict_proba(double *x, int n) {
    double *out = new double[n];

    for (int i = 0; i < n; i++) {
        double score = bias;
        for (int j = 0; j < n_features; j+=8) {
            if (j+8 > n_features) {
                for (int k = j; k < n_features; k++) {
                    score += weights[k]*x[i*n_features+k];
                }
            }
            else {
                __m512d w = _mm512_loadu_pd(&weights[j]);
                __m512d z = _mm512_loadu_pd(&x[i*n_features+j]);
                __m512d y = _mm512_mul_pd(w, z);
                score += _mm512_reduce_add_pd(y);
            }
        }
        out[i] = 1.0/(1.0 + exp(-score));
    }

    return out;
}

unsigned int *logistic_regression::predict(double *x, int n) {
    double *scores = predict_proba(x, n);
    unsigned int *res = new unsigned int[n];
    for (int i = 0; i < n; i++) {
        if (scores[i] < 0.5) res[i] = 0;
        else res[i] = 1;
    }

    return res;
}

double logistic_regression::loss(double *x, unsigned int *y, int n) {
    double *scores = predict_proba(x, n);
    double loss = 0.0;

    for (int i = 0; i < n; i++) {
        if (y[i] == 1) loss += -log(scores[i]);
        else loss += -log(1.0-scores[i]);
    }

    loss = loss/(double)n;

    double w_sum = bias*bias;
    for (int i = 0; i < n_features; i+=8) {
        if (i+8 > n_features) {
            for (int j = i; j < n_features; j++) {
                w_sum += weights[j]*weights[j];
            }
        }
        else {
            __m512d w = _mm512_loadu_pd(&weights[i]);
            __m512d u = _mm512_mul_pd(w, w);
            w_sum += _mm512_reduce_add_pd(u);
        }
    }
    loss += l2_reg*w_sum;

    w_sum = abs(bias);
    for (int i = 0; i < n_features; i+=8) {
        if (i+8 > n_features) {
            for (int j = i; j < n_features; j++) {
                w_sum += abs(weights[j]);
            }
        }
        else {
            __m512d w = _mm512_loadu_pd(&weights[i]);
            __m512d u = _mm512_abs_pd(w);
            w_sum += _mm512_reduce_add_pd(u);
        }
    }
    loss += l1_reg*w_sum;

    return loss;
}

void logistic_regression::update_weights_and_biases(double *x, unsigned int *y, int n) {
    int i = 0;

    while (i < n) {
        int start = i;
        int end = min(i+batch_size, n)-1;
        int m = end-start+1;

        double *scores = predict_proba(x+start*n_features, m);
        
        double *w = new double[n_features];
        for (int k = 0; k < n_features; k++) w[k] = 0.0;

        double b = 0;
        double c = 1.0/(double)m;

        for (int j = start; j < start+m; j++) {
            double err = scores[j-start]-(double)y[j];
            for (int k = 0; k < n_features; k+=8) {
                if (k+8 > n_features) {
                    for (int h = k; h < n_features; h++) {
                        w[h] += x[j*n_features+k]*err;
                    }
                }
                else {
                    __m512d w1 = _mm512_loadu_pd(&w[k]);
                    __m512d z = _mm512_loadu_pd(&x[j*n_features+k]);
                    __m512d e = _mm512_set1_pd(err);
                    __m512d u = _mm512_mul_pd(z, e);
                    w1 = _mm512_add_pd(w1, u);
                    _mm512_storeu_pd(&w[k], w1);
                }
            }
            b += err;
        }

        for (int k = 0; k < n_features; k++) {
            w[k] *= c;
            w[k] += 2*l2_reg*weights[k];
            if (weights[k] > 0) w[k] += l1_reg;
            else if (weights[k] < 0) w[k] -= l1_reg;
            weights[k] -= learning_rate*w[k];
        }

        b *= c;
        b += 2*l2_reg*bias;
        if (bias > 0) b += l1_reg;
        else if (bias < 0) b -= l1_reg;

        bias -= learning_rate*b;
        i += batch_size;
    }
}

void logistic_regression::distribute_data(double *x, unsigned int *y, int n) {
    MPI_Request request = MPI_REQUEST_NULL;

    int h = n/n_process;
    int m = n % n_process;

    for (int p = 1; p < n_process; p++) {
        if (p+1 <= m) {
            int u = h+1;
            MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
            if (y != nullptr) MPI_Isend(y+p*u, u, MPI_UNSIGNED, p, p, comm, &request);
        }
        else {
            if (p <= m) {
                int u = h+1;
                MPI_Isend(x+p*u*n_features, h*n_features, MPI_DOUBLE, p, p, comm, &request);
                if (y != nullptr) MPI_Isend(y+p*u, h, MPI_UNSIGNED, p, p, comm, &request);
            }
            else {
                int u = h;
                MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, comm, &request);
                if (y != nullptr) MPI_Isend(y+p*u, u, MPI_UNSIGNED, p, p, comm, &request);
            }
        }
    }
}

void logistic_regression::fit_non_root(int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g = (rank+1 <= m)?h+1:h;

    double *x = new double[g*n_features];
    unsigned int *y = new unsigned int[g];

    MPI_Recv(x, g*n_features, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
    MPI_Recv(y, g, MPI_UNSIGNED, 0, rank, comm, MPI_STATUS_IGNORE);

    initialize_weights(n);

    int n_epochs = epochs;

    while (n_epochs > 0) {
        update_weights_and_biases(x, y, g);
        double l = loss(x, y, g)*g;

        double *wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, (rank-1)%n_process, 0, comm, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) weights[j] += wb1[j];
        bias += wb1[n_features];
        l += wb1[n_features+1];

        double *wb = new double[n_features+2];
        std::copy(weights, weights+n_features, wb);
        wb[n_features] = bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, (rank+1)%n_process, 0, comm);
        
        wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, (rank-1)%n_process, 1, comm, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) weights[j] = wb1[j];
        bias = wb1[n_features];
        l = wb1[n_features+1];

        wb = new double[n_features+2];
        std::copy(weights, weights+n_features, wb);
        wb[n_features] = bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, (rank+1)%n_process, 1, comm);
        n_epochs--;
    }

    save_model_weights(*this, model_path + "." + std::to_string(rank));
}

void logistic_regression::fit_root(double *x_train, unsigned int *y_train, int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g = (m == 0)?h:h+1;

    distribute_data(x_train, y_train, n);
    initialize_weights(n);

    int n_epochs = epochs;

    while (n_epochs > 0) {
        update_weights_and_biases(x_train, y_train, g);
        double l = loss(x_train, y_train, g)*g;

        double *wb = new double[n_features+2];
        std::copy(weights, weights+n_features, wb);
        wb[n_features] = bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, 1, 0, comm);
        
        double *wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, n_process-1, 0, comm, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) weights[j] = wb1[j]/n_process;
        bias = wb1[n_features]/n_process;
        l = wb1[n_features+1];

        wb = new double[n_features+2];
        std::copy(weights, weights+n_features, wb);
        wb[n_features] = bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, 1, 1, comm);
        
        wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, n_process-1, 1, comm, MPI_STATUS_IGNORE);

        std::cout << "Current Loss = " << l/n << std::endl;
        n_epochs--;
    }

    save_model_weights(*this, model_path + ".0");
}

void logistic_regression::predict_non_root(int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g = (rank+1 <= m)?h+1:h;

    double *x = new double[g*n_features];
    MPI_Recv(x, g*n_features, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);

    load_model_weights(*this, model_path + "." + std::to_string(rank));

    unsigned int *out = predict(x, g);
    MPI_Send(out, g, MPI_UNSIGNED, 0, rank, comm);
}

unsigned int *logistic_regression::predict_root(double *x, int n) {
    int h = n/n_process;
    int m = n % n_process;
    int g = (m == 0)?h:h+1;

    distribute_data(x, nullptr, n);
    load_model_weights(*this, model_path + ".0");

    unsigned int *out = new unsigned int[n];
    int k = 0;

    unsigned int *out_p = predict(x, g);
    std::copy(out_p, out_p+g, out);
    k += g;

    for(int p = 1; p < n_process; p++) {
        int gp = (p+1 <= m)?h+1:h;
        out_p = new unsigned int[gp];
        MPI_Recv(out_p, gp, MPI_UNSIGNED, p, p, comm, MPI_STATUS_IGNORE);
        std::copy(out_p, out_p+gp, out+k);
        k += gp;
    }

    return out;
}

void build_model(
            double *x, 
            unsigned int *y, 
            int n, 
            int n_features, 
            double learning_rate, 
            int epochs, 
            int batch_size, 
            double l1_reg, 
            double l2_reg, 
            std::string model_path) {

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    logistic_regression lr(learning_rate, epochs, batch_size, n_features, l1_reg, l2_reg, model_path, MPI_COMM_WORLD);

    if (rank == 0 && (x == nullptr || y == nullptr)) {
        x = new double[n*n_features];
        y = new unsigned int[n];
        generate(x, y, n, n_features);
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) lr.fit_root(x, y, n);
    else lr.fit_non_root(n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Training duration = " << duration.count() << " ms" << std::endl;
    MPI_Finalize();
}

unsigned int *predict_model(
            double *x, 
            int n, 
            int n_features,
            std::string model_path) {

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    logistic_regression lr(0.0, 0, 0, n_features, 0.0, 0.0, model_path, MPI_COMM_WORLD);

    if (rank == 0 && x == nullptr) {
        x = new double[n*n_features];
        generate(x, nullptr, n, n_features);
    }

    unsigned int *out;
    
    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) out = lr.predict_root(x, n);
    else lr.predict_non_root(n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Prediction duration = " << duration.count() << " ms" << std::endl;
    if (rank == 0) print_arr(out, n, 1);

    MPI_Finalize();
    return out;
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int n_features = atoi(argv[2]);
    double learning_rate = atof(argv[3]);
    int epochs = atoi(argv[4]);
    int batch_size = atoi(argv[5]);
    double l1_reg = atof(argv[6]);
    double l2_reg = atof(argv[7]);
    std::string model_path = argv[8];

    double *x;
    unsigned int *y;

    build_model(x, y, n, n_features, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path);
    
    int m = 100;
    unsigned int *res = predict_model(x, m, n_features, model_path);

    return 0;
}