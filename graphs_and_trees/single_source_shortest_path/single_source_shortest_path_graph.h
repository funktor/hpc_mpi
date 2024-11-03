#ifndef SINGLE_SOURCE_SHORTEST_PATH_GRAPH_H
#define SINGLE_SOURCE_SHORTEST_PATH_GRAPH_H

#include <mpi.h> 
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <deque>
#include <tuple>
#include <map>
#include <fcntl.h>
#include <functional>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <ctime> 
#include <stdbool.h>    // bool type

using namespace std;

std::vector<std::tuple<int, int, double>> generate_random_graph(const int num_nodes, const double prob, const int source);
void distribute_edges(const int *src, const int *dst, const double *dists, const int num_nodes, const int num_edges, const int n_process);
void bellman_ford_mpi(const int source, const int num_nodes, const int num_edges, const int rank, const int n_process);
double *bellman_ford_mpi_root(const int source, const std::vector<std::tuple<int, int, double>> edges, const int num_nodes, const int num_edges, const int n_process);
double *bellman_ford(const int source, const std::vector<std::tuple<int, int, double>> edges, const int num_nodes);

#endif