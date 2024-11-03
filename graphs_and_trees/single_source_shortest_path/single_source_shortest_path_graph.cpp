#include "single_source_shortest_path_graph.h"
using namespace std;

std::vector<std::tuple<int, int, double>> generate_random_graph(const int num_nodes, const double prob, const int source){
    std::vector<std::tuple<int, int, double>> out;
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist_u(0.0, 1.0);
    std::uniform_real_distribution<double> dist_n(0.0, 10000.0);

    int *visited = new int[num_nodes];
    for (int i = 0; i < num_nodes; i++) visited[i] = 0;

    int num_v = 0;

    int i = source;
    while (1) {
        int nxt_source = i;
        for (int j = 0; j < num_nodes; j++) {
            if (i != j) {
                double k = dist_u(engine);
                if (k <= prob) {
                    out.push_back(std::make_tuple(i, j, dist_n(engine)));
                    double h = dist_u(engine);
                    if (h <= 0.5 && visited[j] == 0) nxt_source = j;
                }
            }
        }

        visited[i] = 1;
        num_v += 1;

        if (num_v >= num_nodes) break;
        i = nxt_source;
    }

    return out;
}

void distribute_edges(const int *src, const int *dst, const double *dists, const int num_nodes, const int num_edges, const int n_process) {
    MPI_Request request = MPI_REQUEST_NULL;

    int h = num_edges/n_process;
    int m = num_edges % n_process;

    for (int p = 1; p < n_process; p++) {
        if (p+1 <= m) {
            int u = h+1;
            int *src_dist = new int[2*u];
            std::copy(src+p*u, src+(p+1)*u, src_dist);
            std::copy(dst+p*u, dst+(p+1)*u, src_dist+u);
            MPI_Isend(src_dist, 2*u, MPI_INT, p, p, MPI_COMM_WORLD, &request);
            MPI_Isend(dists+p*u, u, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
        }
        else {
            if (p <= m) {
                int u = h+1;
                int *src_dist = new int[2*h];
                std::copy(src+p*u, src+p*u+h, src_dist);
                std::copy(dst+p*u, dst+p*u+h, src_dist+h);
                MPI_Isend(src_dist, 2*h, MPI_INT, p, p, MPI_COMM_WORLD, &request);
                MPI_Isend(dists+p*u, h, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
            }
            else {
                int u = h;
                int *src_dist = new int[2*u];
                std::copy(src+p*u+m, src+p*u+m+h, src_dist);
                std::copy(dst+p*u+m, dst+p*u+m+h, src_dist+h);
                MPI_Isend(src_dist, 2*u, MPI_INT, p, p, MPI_COMM_WORLD, &request);
                MPI_Isend(dists+p*u+m, u, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &request);
            }
        }
    }
}

void bellman_ford_mpi(const int source, const int num_nodes, const int num_edges, const int rank, const int n_process) {
    double *distance = new double[num_nodes];
    for (int i = 0; i < num_nodes; i++) distance[i] = __DBL_MAX__;
    distance[source] = 0.0;

    int h = num_edges/n_process;
    int m = num_edges % n_process;

    int *src, *dst;
    double *dists;
    int g = h;

    if (rank+1 <= m) {
        g = h+1;
        src = new int[g];
        dst = new int[g];
        dists = new double[g];
        int *src_dist = new int[2*g];
        MPI_Recv(src_dist, 2*g, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(dists, g, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::copy(src_dist, src_dist+g, src);
        std::copy(src_dist+g, src_dist+2*g, dst);
    }
    else {
        src = new int[h];
        dst = new int[h];
        dists = new double[h];
        int *src_dist = new int[2*h];
        MPI_Recv(src_dist, 2*h, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(dists, h, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::copy(src_dist, src_dist+h, src);
        std::copy(src_dist+h, src_dist+2*h, dst);
    }

    for (int i = 0; i < num_nodes-1; i++) {
        for (int j = 0; j < g; j++) {
            int u = src[j];
            int v = dst[j];
            double d = dists[j];
            distance[v] = min(distance[v], distance[u] + d);
        }

        double *d = new double[num_nodes];
        MPI_Recv(d, num_nodes, MPI_DOUBLE, (rank-1)%n_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < num_nodes; j++) distance[j] = min(distance[j], d[j]);
        MPI_Send(distance, num_nodes, MPI_DOUBLE, (rank+1)%n_process, 0, MPI_COMM_WORLD);
        
        d = new double[num_nodes];
        MPI_Recv(d, num_nodes, MPI_DOUBLE, (rank-1)%n_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < num_nodes; j++) distance[j] = min(distance[j], d[j]);
        MPI_Send(distance, num_nodes, MPI_DOUBLE, (rank+1)%n_process, 1, MPI_COMM_WORLD);
    }
}

double *bellman_ford_mpi_root(const int source, const std::vector<std::tuple<int, int, double>> edges, const int num_nodes, const int num_edges, const int n_process) {
    double *distance = new double[num_nodes];
    for (int i = 0; i < num_nodes; i++) distance[i] = __DBL_MAX__;
    distance[source] = 0.0;

    int h = num_edges/n_process;
    int m = num_edges % n_process;
    int g = (m == 0)?h:h+1;

    int *src = new int[num_edges];
    int *dst = new int[num_edges];
    double *dists = new double[num_edges];

    for (int i = 0; i < num_edges; i++) {
        src[i] = std::get<0>(edges[i]);
        dst[i] = std::get<1>(edges[i]);
        dists[i] = std::get<2>(edges[i]);
    }

    distribute_edges(src, dst, dists, num_nodes, num_edges, n_process);

    for (int i = 0; i < num_nodes-1; i++) {
        for (int j = 0; j < g; j++) {
            int u = src[j];
            int v = dst[j];
            double d = dists[j];
            distance[v] = min(distance[v], distance[u] + d);
        }

        MPI_Send(distance, num_nodes, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        
        double *d = new double[num_nodes];
        MPI_Recv(d, num_nodes, MPI_DOUBLE, n_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < num_nodes; j++) distance[j] = min(distance[j], d[j]);

        MPI_Send(distance, num_nodes, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        
        d = new double[num_nodes];
        MPI_Recv(d, num_nodes, MPI_DOUBLE, n_process-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < num_nodes; j++) distance[j] = min(distance[j], d[j]);
    }

    return distance;
}

double *bellman_ford(const int source, const std::vector<std::tuple<int, int, double>> edges, const int num_nodes) {
    double *distance = new double[num_nodes];
    for (int i = 0; i < num_nodes; i++) distance[i] = __DBL_MAX__;
    distance[source] = 0.0;

    for (int i = 0; i < num_nodes-1; i++) {
        for (int j = 0; j < edges.size(); j++) {
            int u = std::get<0>(edges[j]);
            int v = std::get<1>(edges[j]);
            double d = std::get<2>(edges[j]);
            distance[v] = min(distance[v], distance[u] + d);
        }
    }

    return distance;
}

int main(int argc, char *argv[]) {
    int num_nodes = atoi(argv[1]);
    double prob = atof(argv[2]);

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::tuple<int, int, double>> inp;
    int num_edges = 0;

    if (rank == 0) {
        inp = generate_random_graph(num_nodes, prob, 0);
        num_edges = inp.size();
    }

    double *out;

    auto start = std::chrono::high_resolution_clock::now();
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) out = bellman_ford_mpi_root(0, inp, num_nodes, num_edges, size);
    else bellman_ford_mpi(0, num_nodes, num_edges, rank, size);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    if (rank == 0) {
        std::cout << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double *out = bellman_ford(0, inp, num_nodes);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

        std::cout << duration.count() << std::endl;
    }

    MPI_Finalize();
    return 0;
}