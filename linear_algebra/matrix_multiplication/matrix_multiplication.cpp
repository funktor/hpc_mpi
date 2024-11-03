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

void dot_mpi(const int n, const int m, const int p, const int rank, const int size) {
    int h = m/size;

    double *a_r = new double[h*n];
    double *b_r = new double[h*p];
    double *merged = new double[h*(n+p)];

    MPI_Recv(merged, h*(n+p), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::copy(merged, merged+h*n, a_r);
    std::copy(merged+h*n, merged+h*n+h*p, b_r);

    double *out = new double[n*p];
    for (int i = 0; i < n*p; i++) out[i] = 0.0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < n; j++) {
            __m512d c = _mm512_set1_pd(a_r[i*n+j]);
            for (int k = 0; k < p; k+=8) {
                if (k+8 > p) {
                    for (int h = k; h < p; h++) {
                        out[j*p+h] += a_r[i*n+j]*b_r[i*p+h];
                    }
                }
                else {
                    __m512d x = _mm512_loadu_pd(&b_r[i*p+k]);
                    __m512d y = _mm512_loadu_pd(&out[j*p+k]);
                    x = _mm512_mul_pd(x, c);
                    y = _mm512_add_pd(y, x);
                    _mm512_storeu_pd(&out[j*p+k], y);
                }
            }
        }
    }

    for (int stage = 1; stage < size; stage *= 2) {
        if ((rank % stage == 0) && (rank-stage >= 0) && ((rank - stage) % (stage*2) == 0)) {
            MPI_Send(out, n*p, MPI_DOUBLE, rank-stage, rank, MPI_COMM_WORLD);
        }
        else if ((rank % stage == 0) && (rank+stage < size) && (rank % (stage*2) == 0)) {
            double *out_r = new double[n*p];
            MPI_Recv(out_r, n*p, MPI_DOUBLE, rank+stage, rank+stage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < n*p; j++) {
                out[j] += out_r[j];
            }
        }
    }
}

double *dot_mpi_root(const double *a, const double *b, const int n, const int m, const int p, const int size) {
    double *out = new double[n*p];
    for (int i = 0; i < n*p; i++) out[i] = 0.0;

    double *aT = transpose(a, n, m);
    int h = m/size;

    MPI_Request request = MPI_REQUEST_NULL;

    for (int i = h; i < m; i += h) {
        double *merged = new double[h*(n+p)];
        std::copy(aT+i*n, aT+(i+h)*n, merged);
        std::copy(b+i*p, b+(i+h)*p, merged+h*n);
        MPI_Isend(merged, h*(n+p), MPI_DOUBLE, i/h, i/h, MPI_COMM_WORLD, &request);
    }

    for (int i1 = 0; i1 < h; i1++) {
        for (int j = 0; j < n; j++) {
            __m512d c = _mm512_set1_pd(aT[i1*n+j]);
            for (int k = 0; k < p; k+=8) {
                if (k+8 > p) {
                    for (int h = k; h < p; h++) {
                        out[j*p+h] += aT[i1*n+j]*b[i1*p+h];
                    }
                }
                else {
                    __m512d x = _mm512_loadu_pd(&b[i1*p+k]);
                    __m512d y = _mm512_loadu_pd(&out[j*p+k]);
                    x = _mm512_mul_pd(x, c);
                    y = _mm512_add_pd(y, x);
                    _mm512_storeu_pd(&out[j*p+k], y);
                }
            }
        }
    }

    for (int stage = 1; stage < size; stage *= 2) {
        double *out_r = new double[n*p];
        MPI_Recv(out_r, n*p, MPI_DOUBLE, stage, stage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < n*p; j++) {
            out[j] += out_r[j];
        }
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