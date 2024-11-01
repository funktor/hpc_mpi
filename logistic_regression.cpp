#include "logistic_regression.h"
using namespace std;

void generate(double *x, int *y, int n, int m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_n(0.0, 1.0);
    std::uniform_int_distribution<int> dist_u(0, 1);

    for (int i = 0; i < n; i++) {
        if (y != nullptr) y[i] = dist_u(engine);
        for (int j = 0; j < m; j++) {
            x[i*m+j] = dist_n(engine);
        }
    }
}

logistic_regression::logistic_regression(){}

logistic_regression::logistic_regression(
                double learning_rate, 
                int num_epochs, 
                int batch_size, 
                int num_features, 
                double l1_reg,
                double l2_reg){
    
    this->learning_rate = learning_rate;
    this->num_epochs = num_epochs;
    this->batch_size = batch_size;
    this->num_features = num_features;
    this->l1_reg = l1_reg;
    this->l2_reg = l2_reg;
}

logistic_regression::~logistic_regression(){}

void logistic_regression::initialize_weights(int n) {
    weights = new double[num_features];

    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0/n);
    for (int i = 0; i < num_features; i++) weights[i] = dist(engine);
    bias = 1.0/n;
}

double *logistic_regression::predict_proba(double *x_test, int n) {
    double *out = new double[n];

    for (int i = 0; i < n; i++) {
        double score = bias;
        for (int j = 0; j < num_features; j+=8) {
            if (j+8 > num_features) {
                for (int k = j; k < num_features; k++) {
                    score += weights[k]*x_test[i*num_features+k];
                }
            }
            else {
                __m512d w = _mm512_loadu_pd(&weights[j]);
                __m512d x = _mm512_loadu_pd(&x_test[i*num_features+j]);
                __m512d y = _mm512_mul_pd(w, x);
                score += _mm512_reduce_add_pd(y);
            }
        }
        out[i] = 1.0/(1.0 + exp(-score));
    }

    return out;
}

int *logistic_regression::predict(double *x_test, int n) {
    double *scores = predict_proba(x_test, n);
    int *res = new int[n];
    for (int i = 0; i < n; i++) {
        if (scores[i] < 0.5) res[i] = 0;
        else res[i] = 1;
    }

    return res;
}


double logistic_regression::loss(double *x_train, int *y_train, int n) {
    double *scores = predict_proba(x_train, n);
    double loss = 0.0;

    for (int i = 0; i < n; i++) {
        if (y_train[i] == 1) loss += -(double)y_train[i]*log(scores[i]);
        else loss += -(1.0-(double)y_train[i])*log(1.0-scores[i]);
    }

    loss = loss/(double)n;

    double w_sum = bias*bias;
    for (int i = 0; i < num_features; i+=8) {
        if (i+8 > num_features) {
            for (int j = i; j < num_features; j++) {
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
    for (int i = 0; i < num_features; i+=8) {
        if (i+8 > num_features) {
            for (int j = i; j < num_features; j++) {
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

void logistic_regression::update_weights_and_biases(double *x_train, int *y_train, int n) {
    int i = 0;

    while (i < n) {
        int start = i;
        int end = min(i+batch_size, n)-1;
        int m = end-start+1;

        double *scores = predict_proba(x_train+start*num_features, m);
        double *w = new double[num_features];
        double b = 0;
        double c = 1.0/(double)m;

        for (int j = start; j < start+m; j++) {
            double err = scores[j-start]-(double)y_train[j];
            for (int k = 0; k < num_features; k+=8) {
                if (k+8 > num_features) {
                    for (int h = k; h < num_features; h++) {
                        w[h] += x_train[j*num_features+k]*err;
                    }
                }
                else {
                    __m512d w1 = _mm512_loadu_pd(&w[k]);
                    __m512d x = _mm512_loadu_pd(&x_train[j*num_features+k]);
                    __m512d e = _mm512_set1_pd(err);
                    __m512d u = _mm512_mul_pd(x, e);
                    w1 = _mm512_add_pd(w1, u);
                    _mm512_storeu_pd(&w[k], w1);
                }
            }
            b += err;
        }

        for (int k = 0; k < num_features; k++) {
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

void logistic_regression::fit(double *x_train, int *y_train, int n) {
    initialize_weights(n);
    int epochs = num_epochs;

    while (epochs > 0) {
        update_weights_and_biases(x_train, y_train, n);
        double l = loss(x_train, y_train, n);
        std::cout << "Current Loss = " << l << std::endl;
        epochs--;
    }
}

void save_model_weights(logistic_regression &lr, std::string path) {
    std::ofstream outfile(path);
    if (outfile.is_open()) {
        for (int i = 0; i < lr.num_features; i++) outfile << lr.weights[i] << " ";
        outfile << lr.bias;
    }
    outfile.close();
}

void load_model_weights(logistic_regression &lr, std::string path) {
    lr.weights = new double[lr.num_features];

    std::ifstream infile(path);
    if (infile.is_open()) {
        for (int i = 0; i < lr.num_features; i++) infile >> lr.weights[i];
        infile >> lr.bias;
    }
    infile.close();
}

void distribute_data(
        double *x, 
        int *y, 
        int n, 
        int n_features, 
        int n_process) {

    MPI_Request request = MPI_REQUEST_NULL;

    int h = n/n_process;
    int m = n % n_process;

    for (int p = 1; p < n_process; p++) {
        if (p+1 <= m) {
            int u = h+1;
            MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
            if (y != nullptr) MPI_Isend(y+p*u, u, MPI_INT, p, p, MPI_COMM_WORLD, &request);
        }
        else {
            if (p <= m) {
                int u = h+1;
                MPI_Isend(x+p*u*n_features, h*n_features, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
                if (y != nullptr) MPI_Isend(y+p*u, h, MPI_INT, p, p, MPI_COMM_WORLD, &request);
            }
            else {
                int u = h;
                MPI_Isend(x+p*u*n_features, u*n_features, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
                if (y != nullptr) MPI_Isend(y+p*u, u, MPI_INT, p, p, MPI_COMM_WORLD, &request);
            }
        }
    }
}

void lr_train(
        int n, 
        int n_features, 
        int rank, 
        int n_process, 
        double learning_rate, 
        int epochs, 
        int batch_size, 
        double l1_reg, 
        double l2_reg, 
        std::string model_path) {

    int h = n/n_process;
    int m = n % n_process;
    int g = (rank+1 <= m)?h+1:h;

    double *x = new double[g*n_features];
    int *y = new int[g];

    MPI_Recv(x, g*n_features, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(y, g, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    logistic_regression lr(learning_rate, epochs, batch_size, n_features, 0.0, 0.0);

    lr.initialize_weights(n);

    while (epochs > 0) {
        lr.update_weights_and_biases(x, y, g);
        double l = lr.loss(x, y, g)*g;

        double *wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, (rank-1)%n_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) lr.weights[j] += wb1[j];
        lr.bias += wb1[n_features];
        l += wb1[n_features+1];

        double *wb = new double[n_features+2];
        std::copy(lr.weights, lr.weights+n_features, wb);
        wb[n_features] = lr.bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, (rank+1)%n_process, 0, MPI_COMM_WORLD);
        
        wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, (rank-1)%n_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) lr.weights[j] = wb1[j];
        lr.bias = wb1[n_features];
        l = wb1[n_features+1];

        wb = new double[n_features+2];
        std::copy(lr.weights, lr.weights+n_features, wb);
        wb[n_features] = lr.bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, (rank+1)%n_process, 1, MPI_COMM_WORLD);
        epochs--;
    }

    save_model_weights(lr, model_path + "." + std::to_string(rank));
}

void lr_train_root(
        double *x, 
        int *y, 
        int n, 
        int n_features, 
        int n_process, 
        double learning_rate, 
        int epochs, 
        int batch_size, 
        double l1_reg, 
        double l2_reg, 
        std::string model_path) {

    int h = n/n_process;
    int m = n % n_process;
    int g = (m == 0)?h:h+1;

    distribute_data(x, y, n, n_features, n_process);

    logistic_regression lr(learning_rate, epochs, batch_size, n_features, 0.0, 0.0);

    lr.initialize_weights(n);

    while (epochs > 0) {
        lr.update_weights_and_biases(x, y, g);
        double l = lr.loss(x, y, g)*g;

        double *wb = new double[n_features+2];
        std::copy(lr.weights, lr.weights+n_features, wb);
        wb[n_features] = lr.bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        
        double *wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, n_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < n_features; j++) lr.weights[j] = wb1[j]/n_process;
        lr.bias = wb1[n_features]/n_process;
        l = wb1[n_features+1];

        wb = new double[n_features+2];
        std::copy(lr.weights, lr.weights+n_features, wb);
        wb[n_features] = lr.bias;
        wb[n_features+1] = l;

        MPI_Send(wb, n_features+2, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        
        wb1 = new double[n_features+2];
        MPI_Recv(wb1, n_features+2, MPI_DOUBLE, n_process-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout << "Current Loss = " << l/n << std::endl;
        epochs--;
    }

    save_model_weights(lr, model_path + ".0");
}

void lr_predict(
        int n, 
        int n_features, 
        int rank, 
        int n_process, 
        std::string model_path) {

    int h = n/n_process;
    int m = n % n_process;
    int g = (rank+1 <= m)?h+1:h;

    double *x = new double[g*n_features];
    MPI_Recv(x, g*n_features, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    logistic_regression lr(0.0, 0, 0, n_features, 0.0, 0.0);
    load_model_weights(lr, model_path + "." + std::to_string(rank));

    int *out = lr.predict(x, g);
    MPI_Send(out, g, MPI_INT, 0, rank, MPI_COMM_WORLD);
}

int *lr_predict_root(
        double *x, 
        int n, 
        int n_features, 
        int n_process, 
        std::string model_path) {

    int h = n/n_process;
    int m = n % n_process;
    int g = (m == 0)?h:h+1;

    distribute_data(x, nullptr, n, n_features, n_process);

    logistic_regression lr(0.0, 0, 0, n_features, 0.0, 0.0);
    load_model_weights(lr, model_path + ".0");

    int *out = new int[n];
    int k = 0;

    int *out_self = lr.predict(x, g);
    std::copy(out_self, out_self+g, out);
    k += g;

    for(int p = 1; p < n_process; p++) {
        int gp = (p+1 < m)?h+1:h;
        int *out_p = new int[gp];
        MPI_Recv(out_p, gp, MPI_INT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::copy(out_p, out_p+gp, out+k);
        k += gp;
    }

    return out;
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

void build_model(
            double *x, 
            int *y, 
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

    if (rank == 0 && (x == nullptr || y == nullptr)) {
        x = new double[n*n_features];
        y = new int[n];
        generate(x, y, n, n_features);
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) lr_train_root(x, y, n, n_features, size, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path);
    else lr_train(n, n_features, rank, size, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Training duration = " << duration.count() << " ms" << std::endl;
    MPI_Finalize();
}

int *predict_model(
            double *x, 
            int n, 
            int n_features,
            std::string model_path) {

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && x == nullptr) {
        x = new double[n*n_features];
        generate(x, nullptr, n, n_features);
    }

    int *out;
    
    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) out = lr_predict_root(x, n, n_features, size, model_path);
    else lr_predict(n, n_features, rank, size, model_path);
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
    int *y;

    build_model(x, y, n, n_features, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path);
    
    int m = 100;
    int *res = predict_model(x, m, n_features, model_path);

    return 0;
}