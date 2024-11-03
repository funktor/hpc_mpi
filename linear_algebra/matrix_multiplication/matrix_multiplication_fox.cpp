#include "matrix_multiplication.h"
using namespace std;

void generate(double *inp, int n, int m){
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            inp[i*m+j] = dist(engine);
        }
    }
}

double *transpose(const double *a, const int n, const int m) {
    double *b = new double[n*m];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[j*n+i] = a[i*m+j];
        }
    }

    return b;
}

void print_arr(const double *arr, const int n, const int m) {
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

void dot_mpi(const int n, const int m, const int p, const int rank, const int n_process) {
    int h = n/n_process;

    double *a_curr = new double[h];
    double *b_curr;

    if (rank == n_process-1) {
        int cnt = h*p + (m % h)*p;
        b_curr = new double[cnt];
        MPI_Recv(b_curr, cnt, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
        b_curr = new double[h*p];
        MPI_Recv(b_curr, h*p, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    double *out = new double[h*p];
    for (int i = 0; i < h*p; i++) out[i] = 0.0;

    for (int stage = 0; stage < m; stage++) {
        MPI_Recv(a_curr, h, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < p; j++) out[i*p+j] += a_curr[i]*b_curr[i*p+j];
        }

        double *b_shift = new double[p];

        MPI_Recv(b_shift, p, MPI_DOUBLE, (rank+1) % n_process, (rank+1) % n_process, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(b_curr, p, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD);
        
        double *b_curr_new;

        if (rank == n_process-1) {
            int cnt = h*p + (m % h)*p;
            b_curr_new = new double[cnt];
            std::copy(b_curr+p, b_curr+cnt, b_curr_new);
            std::copy(b_shift, b_shift+p, b_curr_new+cnt-p);
            b_curr = b_curr_new;
        }
        else {
            b_curr_new = new double[h*p];
            std::copy(b_curr+p, b_curr+h*p, b_curr_new);
            std::copy(b_shift, b_shift+p, b_curr_new+(h-1)*p);
            b_curr = b_curr_new;
        }
    }

    MPI_Send(out, h*p, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
}

double *dot_mpi_root(const double *a, const double *b, const int n, const int m, const int p, const int n_process) {
    int h = n/n_process;

    double *out = new double[n*p];
    for (int i = 0; i < n*p; i++) out[i] = 0.0;

    double *a_curr = new double[h];
    double *b_curr;

    MPI_Request request = MPI_REQUEST_NULL;

    if (n_process > 1) {
        b_curr = new double[h*p];
        int h_1 = h;
        int g_1 = 0;
        while (h_1 > m) {
            std::copy(b, b+m*p, b_curr+g_1);
            g_1 += m*p;
            h_1 -= m;
        }
        if (h_1 > 0) std::copy(b, b+h_1*p, b_curr+g_1);
    }
    else {
        b_curr = new double[h*p+(m % h)*p];
        int h_1 = h;
        int g_1 = 0;
        while (h_1 > m) {
            std::copy(b, b+m*p, b_curr+g_1);
            g_1 += m*p;
            h_1 -= m;
        }
        std::copy(b, b+m*p, b_curr+g_1);
    }
    

    for (int i = h; i < n; i += h) {
        int j = i % m;

        if (i/h == n_process-1) {
            if (j+h > m) {
                double *b_arr = new double[h*p + (m % h)*p];
                std::copy(b+j*p, b+m*p, b_arr);

                int h_1 = h-(m-j);
                int g_1 = (m-j)*p;
                while (h_1 > m) {
                    std::copy(b, b+m*p, b_arr+g_1);
                    g_1 += m*p;
                    h_1 -= m;
                }
                std::copy(b, b+m*p, b_arr+g_1);
                MPI_Isend(b_arr, h*p + (m % h)*p, MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
            }
            else {
                MPI_Isend(&b[j*p], (m-j)*p, MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
            }
        }
        else {
            if (j+h > m) {
                double *b_arr = new double[h*p];
                std::copy(b+j*p, b+m*p, b_arr);

                int h_1 = h-(m-j);
                int g_1 = (m-j)*p;
                while (h_1 > m) {
                    std::copy(b, b+m*p, b_arr+g_1);
                    g_1 += m*p;
                    h_1 -= m;
                }
                if (h_1 > 0) std::copy(b, b+h_1*p, b_arr+g_1);
                MPI_Isend(b_arr, h*p, MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
            }
            else {
                MPI_Isend(&b[j*p], h*p, MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
            }
        }
    }

    for (int stage = 0; stage < m; stage++) {
        for (int j = 0; j < h; j++) a_curr[j] = a[j*m+(j+stage)%m];

        for (int i = h; i < n; i += h) {
            double *a_arr = new double[h];
            for (int j = i; j < i+h; j++) a_arr[j-i] = a[j*m+(j+stage)%m];
            MPI_Isend(a_arr, h, MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < p; j++) out[i*p+j] += a_curr[i]*b_curr[i*p+j];
        }

        double *b_shift = new double[p];

        if (n_process > 1) {
            MPI_Send(b_curr, p, MPI_DOUBLE, n_process-1, 0, MPI_COMM_WORLD);
            MPI_Recv(b_shift, p, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            std::copy(b_curr, b_curr+p, b_shift);
        }

        double *b_curr_new;

        if (n_process == 1) {
            int cnt = h*p + (m % h)*p;
            b_curr_new = new double[cnt];
            std::copy(b_curr+p, b_curr+cnt, b_curr_new);
            std::copy(b_shift, b_shift+p, b_curr_new+cnt-p);
            b_curr = b_curr_new;
        }
        else {
            b_curr_new = new double[h*p];
            std::copy(b_curr+p, b_curr+h*p, b_curr_new);
            std::copy(b_shift, b_shift+p, b_curr_new+(h-1)*p);
            b_curr = b_curr_new;
        }
    }

    for (int process = 1; process < n_process; process++) {
        MPI_Recv(&out[process*h*p], h*p, MPI_DOUBLE, process, process, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return out;
}

double *dot(const double *a, const double *b, const int n, const int m, const int p) {
    double *out = new double[n*p];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) {
                out[i*p+k] += a[i*m+j]*b[j*p+k];
            }
        }
    }

    return out;
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int p = atoi(argv[3]);

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *a, *b;

    if (rank == 0) {
        a = new double[n*m];
        b = new double[m*p];

        generate(a, n, m);
        generate(b, m, p);
    }

    double *out;

    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) out = dot_mpi_root(a, b, n, m, p, size);
    else dot_mpi(n, m, p, rank, size);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    if (rank == 0) {
        std::cout << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double *out = dot(a, b, n, m, p);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

        std::cout << duration.count() << std::endl;
    }

    MPI_Finalize();
    return 0;
}