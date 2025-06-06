#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
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
#include <fstream>
#include <cmath>
#include <variant>
#include <omp.h>

using namespace std;

template<typename T>
class NodeFunc;

template<typename T>
class Node;

template<typename T>
class Variable;

class Constant;
class Input;
class Graph;

std::string get_uuid() {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_int_distribution<int> dist(0, 15);

    const char *v = "0123456789abcdef";
    const bool dash[] = { 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 };

    std::string res;
    for (int i = 0; i < 16; i++) {
        if (dash[i]) res += "-";
        res += v[dist(rng)];
        res += v[dist(rng)];
    }
    return res;
}

void print_vector(std::vector<double> x, size_t n) {
    std::cout << "[";
    for (auto i = 0; i < std::min(n, x.size()); i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

void print_vector(double *x, size_t n) {
    std::cout << "[";
    for (auto i = 0; i < n; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

void print_vector(unsigned int *x, size_t n) {
    std::cout << "[";
    for (auto i = 0; i < n; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

struct Tensor {
    double *values;
    unsigned int *shape;
    unsigned int n_dim;
    bool is_identity=false;
    std::unordered_map<unsigned long, double> identity_vals;
};

template<typename T>
class NodeFunc {
    private:
    public:
        std::string id;
        bool is_param = false;
        bool is_input = false;
        bool is_output = false;
        bool is_constant = false;
        Tensor* param_val;
        Tensor* constant_val;
        std::function<Tensor*()> func;
        std::function<Tensor**()> d_func;
        NodeFunc** children;
        unsigned long num_children = 0;
        unsigned int n_dim = 0;

        NodeFunc(){
            id = get_uuid();
        }

        bool operator==(const NodeFunc *other) const { 
            return this->id == other->id;
        }
};

template <typename T>
struct std::hash<NodeFunc<T>> {
  std::size_t operator()(const NodeFunc<T> *k) const {
    return std::hash<std::string>()(k->toString());
  }
};

struct LevelData {
    std::vector<NodeFunc<double>*> curr_level_nodes;
};

struct GradCell {
    NodeFunc<double>* src;
    NodeFunc<double>* dst;

    bool operator==(const GradCell &other) const { 
        return src->id == other.src->id && dst->id == other.dst->id;
    }

    std::string toString() const {
        return src->id + " " + dst->id;
    }
};

template <>
struct std::hash<GradCell> {
  std::size_t operator()(const GradCell& k) const {
    return std::hash<std::string>()(k.toString());
  }
};

unsigned long get_tensor_n_elements(Tensor *a) {
    unsigned long n = 1;
    for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];
    return n;
}

unsigned long get_tensor_n_elements_wo_first(Tensor *a) {
    unsigned long n = 1;
    for (auto i = 1; i < a->n_dim; i++) n *= a->shape[i];
    return n;
}

bool is_same_shape(Tensor *a, Tensor *b) {
    if (a->n_dim == b->n_dim) {
        for (auto i = 0; i < a->n_dim; i++) {
            if (a->shape[i] != b->shape[i]) return false;
        }
        return true;
    }
    return false;
}

Tensor *add(Tensor *a, Tensor *b) {
    omp_set_num_threads(4);
    Tensor *out = new Tensor();

    if (is_same_shape(a, b)) {
        out->n_dim = a->n_dim;
        out->shape = new unsigned int[out->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, out->shape);

        unsigned long n = get_tensor_n_elements(a);
        out->values = new double[n];

        #pragma omp parallel for shared(a, b, out)
        for (auto i = 0; i < n; i++) out->values[i] = a->values[i] + b->values[i];

        return out;
    }

    return nullptr;
}

Tensor *dotp(Tensor *a, Tensor *b, bool b_param) {
    omp_set_num_threads(8);

    if (b_param) {
        Tensor *out = new Tensor();
        out->n_dim = a->n_dim;

        out->shape = new unsigned int[a->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, out->shape);

        out->shape[1] = b->shape[1];

        if (a->n_dim == 2 && b->n_dim == 2 && a->shape[1] == b->shape[0]) {
            unsigned int n = out->shape[0]*out->shape[1];
            out->values = new double[n];

            #pragma omp parallel for shared(out)
            for (auto i = 0; i < n; i++) out->values[i] = 0.0;

            #pragma omp parallel for shared(a, b, out)
            for (auto i = 0; i < a->shape[0]; i++) {
                for (auto j = 0; j < a->shape[1]; j++) {
                    for (auto k = 0; k < b->shape[1]; k++) {
                        out->values[i*b->shape[1]+k] += a->values[i*a->shape[1]+j]*b->values[j*b->shape[1]+k];
                    }
                }
            }
        }

        return out;
    }
    else {
        Tensor *out = new Tensor();
        out->n_dim = a->n_dim;

        out->shape = new unsigned int[a->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, out->shape);

        if (a->n_dim == 2 && b->n_dim == 3 and a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]) {
            out->shape[1] = b->shape[2];
            unsigned int n = out->shape[0]*out->shape[1];
            out->values = new double[n];

            #pragma omp parallel for shared(out)
            for (auto i = 0; i < n; i++) out->values[i] = 0.0;

            if (b->is_identity) {
                #pragma omp parallel for shared(a, b, out)
                for (auto q = 0; q < a->shape[0]; q++) {
                    for (auto j = 0; j < a->shape[1]; j++) {
                        out->values[q*b->shape[2]+j] += a->values[q*a->shape[1]+j]*b->identity_vals[q*b->shape[1]*b->shape[2]+j*b->shape[2]+j];
                    }
                }
            }
            else {
                #pragma omp parallel for shared(a, b, out)
                for (auto q = 0; q < a->shape[0]; q++) {
                    for (auto j = 0; j < a->shape[1]; j++) {
                        for (auto k = 0; k < b->shape[2]; k++) {
                            out->values[q*b->shape[2]+k] += a->values[q*a->shape[1]+j]*b->values[q*b->shape[1]*b->shape[2]+j*b->shape[2]+k];
                        }
                    }
                }
            }
        }
        else if (a->n_dim == 1 && b->n_dim == 2 and a->shape[0] == b->shape[0]) {
            out->shape[0] = b->shape[1];

            unsigned int n = out->shape[0];
            out->values = new double[n];

            #pragma omp parallel for shared(out)
            for (auto i = 0; i < n; i++) out->values[i] = 0.0;

            #pragma omp parallel for shared(a, b, out)
            for (auto i = 0; i < a->shape[0]; i++) {
                for (auto j = 0; j < b->shape[1]; j++) {
                    out->values[j] += a->values[i]*b->values[i*b->shape[1]+j];
                }
            }
        }

        return out;
    }
}

unsigned int *dotp_shape(Tensor *a, Tensor *b, bool b_param) {
    if (b_param) {
        unsigned int *shape = new unsigned int[a->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, shape);
        shape[1] = b->shape[1];

        return shape;
    }
    else {
        unsigned int *shape = new unsigned int[a->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, shape);
        shape[1] = b->shape[2];

        return shape;
    }
}

class Graph {
    private:
    public:
        std::unordered_map<std::string, Tensor*> func_cached;
        std::unordered_map<std::string, Tensor**> d_func_cached;
        std::vector<LevelData*> dag;
        std::unordered_map<GradCell, Tensor*> grad_acc;
        unsigned int batch_size = 1;
        NodeFunc<double> *root_node;

        Graph() {}

        NodeFunc<double> *_constant(double init_v, unsigned int *shape, unsigned int shape_dim) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_constant = true;

            Tensor *g = new Tensor();
            g->n_dim = shape_dim;
            g->shape = new unsigned int[shape_dim];
            std::copy(shape, shape+shape_dim, g->shape);

            unsigned int n = 1;
            for (auto i = 0; i < shape_dim; i++) n *= shape[i];

            g->values = new double[n];
            for (auto i = 0; i < n; i++) g->values[i] = init_v;

            obj->constant_val = g;
            std::string id = obj->id;

            obj->func = [obj](){return obj->constant_val;};
            obj->d_func = [this, id, obj](){                
                Tensor **out = new Tensor*[1];
                out[0] = new Tensor();
                out[0]->n_dim = 3;

                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < obj->constant_val->n_dim; i++) m *= obj->constant_val->shape[i];

                out[0]->shape[0] = obj->constant_val->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->values = new double[obj->constant_val->shape[0]*m*m];
                for (auto j = 0; j < obj->constant_val->shape[0]*m*m; j++) out[0]->values[j] = 0.0;

                return out;
            };

            return obj;
        }

        NodeFunc<double> *_input(unsigned int *shape, unsigned int shape_dim) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_input = true;

            Tensor *g = new Tensor();
            g->n_dim = shape_dim;

            g->shape = new unsigned int[shape_dim];
            std::copy(shape, shape+shape_dim, g->shape);

            unsigned int n = 1;
            for (auto i = 0; i < shape_dim; i++) n *= shape[i];

            g->values = new double[n];

            obj->constant_val = g;
            std::string id = obj->id;

            obj->func = [obj](){return obj->constant_val;};
            obj->d_func = [this, id, obj](){                
                Tensor **out = new Tensor*[1];
                out[0] = new Tensor();
                out[0]->n_dim = 3;

                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < obj->constant_val->n_dim; i++) m *= obj->constant_val->shape[i];

                out[0]->shape[0] = obj->constant_val->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->values = new double[obj->constant_val->shape[0]*m*m];
                for (auto j = 0; j < obj->constant_val->shape[0]*m*m; j++) out[0]->values[j] = 0.0;

                return out;
            };

            return obj;
        }

        NodeFunc<double> *_output(unsigned int *shape, unsigned int shape_dim) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_output = true;

            Tensor *g = new Tensor();
            g->n_dim = shape_dim;
            g->shape = new unsigned int[shape_dim];
            std::copy(shape, shape+shape_dim, g->shape);

            unsigned int n = 1;
            for (auto i = 0; i < shape_dim; i++) n *= shape[i];
            g->values = new double[n];

            obj->constant_val = g;
            std::string id = obj->id;

            obj->func = [obj](){return obj->constant_val;};
            obj->d_func = [this, id, obj](){                
                Tensor **out = new Tensor*[1];
                out[0] = new Tensor();
                out[0]->n_dim = 3;

                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < obj->constant_val->n_dim; i++) m *= obj->constant_val->shape[i];

                out[0]->shape[0] = obj->constant_val->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->values = new double[obj->constant_val->shape[0]*m*m];
                for (auto j = 0; j < obj->constant_val->shape[0]*m*m; j++) out[0]->values[j] = 0.0;

                return out;
            };

            return obj;
        }

        NodeFunc<double> *_parameter(double *init_v, unsigned int *shape, unsigned int shape_dim) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_param = true;

            Tensor *g = new Tensor();
            g->n_dim = shape_dim;
            g->shape = new unsigned int[shape_dim];
            std::copy(shape, shape+shape_dim, g->shape);

            unsigned int n = 1;
            for (auto i = 0; i < shape_dim; i++) n *= shape[i];

            g->values = new double[n];
            for (auto i = 0; i < n; i++) g->values[i] = init_v[i];

            obj->param_val = g;
            std::string id = obj->id;

            obj->func = [obj](){return obj->param_val;};
            obj->d_func = [this, id, obj](){                
                Tensor **out = new Tensor*[1];
                out[0] = new Tensor();
                out[0]->n_dim = 3;

                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < obj->param_val->n_dim; i++) m *= obj->param_val->shape[i];

                out[0]->shape[0] = obj->param_val->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->values = new double[obj->param_val->shape[0]*m*m];
                for (auto j = 0; j < obj->param_val->shape[0]*m*m; j++) out[0]->values[j] = 0.0;

                return out;
            };

            return obj;
        }

        NodeFunc<double> *_add(NodeFunc<double> *inp1, NodeFunc<double> *inp2) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp1, inp2, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;

                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = a->values[i] + b->values[i];

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp1, inp2, this, id](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = 1.0;
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 3;
                out[1]->shape = new unsigned int[3];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;
                out[1]->shape[2] = m;

                out[1]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[1]->identity_vals[i*m*m+j*m+j] = 1.0;
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }
        
        NodeFunc<double> *_sub(NodeFunc<double> *inp1, NodeFunc<double> *inp2) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp1, inp2, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = a->values[i] - b->values[i];

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp1, inp2, this, id](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = 1.0;
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 3;
                out[1]->shape = new unsigned int[3];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;
                out[1]->shape[2] = m;

                out[1]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[1]->identity_vals[i*m*m+j*m+j] = -1.0;
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }
        
        NodeFunc<double> *_mul(NodeFunc<double> *inp1, NodeFunc<double> *inp2) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp1, inp2, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = a->values[i] * b->values[i];

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp1, inp2, this, id](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = b->values[i*m+j];
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 3;
                out[1]->shape = new unsigned int[3];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;
                out[1]->shape[2] = m;

                out[1]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[1]->identity_vals[i*m*m+j*m+j] = a->values[i*m+j];
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }

        NodeFunc<double> *_dot(NodeFunc<double> *inp1, NodeFunc<double> *inp2) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp1, inp2, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out = dotp(a, b, true);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp1, inp2, this, id](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                unsigned int *shape = dotp_shape(a, b, true);

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= shape[i];

                unsigned int n_a = 1;
                unsigned int n_b = 1;
                for (auto i = 1; i < a->n_dim; i++) n_a *= a->shape[i];
                for (auto i = 0; i < b->n_dim; i++) n_b *= b->shape[i];

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = n_a;

                out[0]->values = new double[a->shape[0]*m*n_a];

                omp_set_num_threads(8);
                #pragma omp parallel for shared(a, b, out)
                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        for (auto k = 0; k < n_a; k++) {
                            out[0]->values[i*m*n_a+j*n_a+k] = b->values[k*m+j];
                        }
                    }
                }


                out[1] = new Tensor();
                
                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }
        
        NodeFunc<double> *_sin(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = sin(a->values[i]);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = cos(a->values[i*m+j]);
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }
        
        NodeFunc<double> *_cos(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = cos(a->values[i]);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = -sin(a->values[i*m+j]);
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }
        
        NodeFunc<double> *_pow(NodeFunc<double> *inp, double p) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id, p](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = pow(a->values[i], p);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id, p](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = p*pow(a->values[i*m+j], p-1);
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }
        
        NodeFunc<double> *_exp(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = exp(a->values[i]);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = exp(a->values[i*m+j]);
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }
        
        NodeFunc<double> *_log(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = log(a->values[i]);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = pow(a->values[i*m+j], -1.0);
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<double> *_relu(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = (a->values[i] > 0.0)?a->values[i]:0.0;

                func_cached[id] = out;

                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->identity_vals[i*m*m+j*m+j] = (a->values[i*m+j] > 0.0)?1.0:0.0;
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<double> *_sigmoid(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < n; i++) out->values[i] = 1.0/(1.0+exp(-a->values[i]));

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->is_identity = true;

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        double w = 1.0/(1.0+exp(-a->values[i*m+j]));
                        out[0]->identity_vals[i*m*m+j*m+j] = w*(1.0-w);
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }


        NodeFunc<double> *_softmax(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                double *maxv = new double[a->shape[0]];
                double *sumv = new double[a->shape[0]];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                for (auto i = 0; i < a->shape[0]; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < a->shape[0]; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                out->n_dim = a->n_dim;
                out->shape = new unsigned int[a->n_dim];
                std::copy(a->shape, a->shape+a->n_dim, out->shape);

                out->values = new double[n];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) out->values[i*m+j] = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                }

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, this, id](){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                double *maxv = new double[a->shape[0]];
                double *sumv = new double[a->shape[0]];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                for (auto i = 0; i < a->shape[0]; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < a->shape[0]; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                out[0] = new Tensor();
                out[0]->n_dim = 3;
                out[0]->shape = new unsigned int[3];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->shape[2] = m;

                out[0]->values = new double[a->shape[0]*m*m];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        double z1 = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                        for (auto k = 0; k < m; k++) {
                            double z2 = exp(a->values[i*m+k]-maxv[i])/sumv[i];
                            out[0]->values[i*m*m+j*m+k] = (j == k)?z1*(1.0-z1):-z1*z2;
                        }
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<double> *_mse_loss(NodeFunc<double> *inp, NodeFunc<double> *oup) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, oup, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;
                out->values = new double[1]; 
                out->values[0] = 0.0;

                for (auto i = 0; i < n; i++) out->values[0] += 0.5*(a->values[i]-b->values[i])*(a->values[i]-b->values[i]);

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [this, id, inp, oup](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;

                out[0]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->values[i*m+j] = a->values[i*m+j]-b->values[i*m+j];
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                m = 1;
                for (auto i = 1; i < b->n_dim; i++) m *= b->shape[i];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;

                out[1]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < b->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[1]->values[i*m+j] = -(a->values[i*m+j]-b->values[i*m+j]);
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<double> *_logistic_loss(NodeFunc<double> *inp, NodeFunc<double> *oup) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, oup, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;
                out->values = new double[1];
                out->values[0] = 0.0;

                for (auto j = 0; j < n; j++) {
                    if (a->values[j] > 0.0 && a->values[j] < 1.0) out->values[0] += -b->values[j]*log(a->values[j])-(1.0-b->values[j])*log(1.0-a->values[j]);
                    else if (a->values[j] <= 0.0) out->values[0] += -(1.0-b->values[j])*log(1.0-a->values[j]);
                    else if (a->values[j] >= 1.0) out->values[0] += -b->values[j]*log(a->values[j]);
                }

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [this, id, inp, oup](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;
                out[0]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->values[i*m+j] = (a->values[i*m+j] == 0.0 || a->values[i*m+j] == 1.0)?0.0:(a->values[i*m+j]-b->values[i*m+j])/(a->values[i*m+j]*(1.0-a->values[i*m+j]));
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                m = 1;
                for (auto i = 1; i < b->n_dim; i++) m *= b->shape[i];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;
                out[1]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < b->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (a->values[i*m+j] > 0.0 && a->values[i*m+j] < 1.0) out[1]->values[i*m+j] = -log(a->values[i*m+j])+log(1-a->values[i*m+j]);
                        else if (a->values[i*m+j] <= 0.0) out[1]->values[i*m+j] = log(1-a->values[i*m+j]);
                        else if (a->values[i*m+j] >= 1.0) out[1]->values[i*m+j] = -log(a->values[i*m+j]);
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<double> *_cross_entropy_loss(NodeFunc<double> *inp, NodeFunc<double> *oup) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;

            obj->func = [inp, oup, this, id](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                unsigned int n = 1;
                for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;
                out->values = new double[1];
                out->values[0] = 0.0;

                for (auto j = 0; j < n; j++) out->values[0] += (a->values[j] > 0.0)?-b->values[j]*log(a->values[j]):0.0;

                func_cached[id] = out;
                return out;
            };

            obj->d_func = [this, id, inp, oup](){
                Tensor **out = new Tensor*[2];
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int m = 1;
                for (auto i = 1; i < a->n_dim; i++) m *= a->shape[i];

                out[0]->shape[0] = a->shape[0];
                out[0]->shape[1] = m;

                out[0]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[0]->values[i*m+j] = (a->values[i*m+j] != 0.0)?-b->values[i*m+j]*pow(a->values[i*m+j], -1.0):0.0;
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                m = 1;
                for (auto i = 1; i < b->n_dim; i++) m *= b->shape[i];

                out[1]->shape[0] = a->shape[0];
                out[1]->shape[1] = m;

                out[1]->values = new double[a->shape[0]*m];

                for (auto i = 0; i < a->shape[0]; i++) {
                    for (auto j = 0; j < m; j++) {
                        out[1]->values[i*m+j] = (a->values[i*m+j] > 0.0)?-log(a->values[i*m+j]):0.0;
                    }
                }
                
                return out;
            };

            obj->children = new NodeFunc<double>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        void build_dag() {
            struct DQ {
                NodeFunc<double> *node;
                unsigned int level;

                std::string toString() const {
                    return node->id + " " + std::to_string(level);
                }

                bool operator==(const DQ& other) const {
                    return other.toString() == this->toString();
                }
            };

            struct DQ_hash {
                std::size_t operator()(const DQ& k) const {
                  return std::hash<std::string>()(k.toString());
                }
            };

            std::deque<DQ> dq;
            std::vector<NodeFunc<double>*> curr_level_nodes;
            std::unordered_set<DQ, DQ_hash> added_nodes;

            dq.push_back({root_node, 1});
            added_nodes.insert({root_node, 1});
            
            unsigned int curr_level = 0;

            while (dq.size() > 0) {
                DQ x = dq.front();
                dq.pop_front();

                NodeFunc<double> *curr_inp = x.node;
                unsigned int level = x.level;

                if (level > curr_level) {
                    if (curr_level_nodes.size() > 0) {
                        LevelData *data = new LevelData();
                        for (auto nd : curr_level_nodes) {
                            if (!nd->is_param) data->curr_level_nodes.push_back(nd);
                        }
                        dag.push_back(data);
                        curr_level_nodes.clear();
                    }
                    
                    curr_level = level;
                }

                curr_level_nodes.push_back(curr_inp);

                for (auto i = 0; i < curr_inp->num_children; i++) {
                    DQ mydq = {curr_inp->children[i], level+1};

                    if (added_nodes.count(mydq) == 0) {
                        dq.push_back(mydq);
                        added_nodes.insert(mydq);
                    }
                }
            }

            if (curr_level_nodes.size() > 0) {
                LevelData *data = new LevelData();
                for (auto nd : curr_level_nodes) {
                    if (!nd->is_param) data->curr_level_nodes.push_back(nd);
                }
                dag.push_back(data);
            }
        }

        void forward_pass(Tensor *x, Tensor *y) {
            for (int i = dag.size()-1; i >= 0; i--) {
                LevelData* data = dag[i];
                std::vector<NodeFunc<double>*> nodes = data->curr_level_nodes;

                for (auto nd : nodes) {
                    if (nd->is_input) nd->constant_val = x;
                    else if (nd->is_output) nd->constant_val = y;
                    else if (nd->is_constant) nd->constant_val->shape[0] = x->shape[0];

                    nd->func();
                }
            }
        }

        void reset_caches() {
            for (auto kv : func_cached) {
                Tensor *v = kv.second;

                delete [] v->shape;
                delete [] v->values;
                delete v;
            }

            func_cached.clear();
        }
        
        void gradient_accumulation() {
            struct KVPair {
                NodeFunc<double>* node;
                Tensor *value;
            };

            std::unordered_map<GradCell, Tensor*> prev_arr;

            for (auto i = 0; i < dag.size(); i++) {
                LevelData* data = dag[i];

                std::vector<NodeFunc<double>*> nodes = data->curr_level_nodes;
                std::unordered_map<GradCell, Tensor*> curr_grad_map;              

                for (auto k = 0; k < nodes.size(); k++) {
                    if (nodes[k]->num_children > 0) {
                        NodeFunc<double> **children = nodes[k]->children;
                        Tensor **dfunc = nodes[k]->d_func();

                        for (auto j = 0; j < nodes[k]->num_children; j++) {
                            GradCell u = {nodes[k], children[j]};
                            if (!children[j]->is_param) curr_grad_map[u] = dfunc[j];
                        }
                    }
                }

                if (prev_arr.size() > 0) {
                    std::unordered_map<GradCell, Tensor*> new_arr;
                    std::unordered_map<NodeFunc<double>*, std::vector<KVPair>> prev_arr_map;

                    for (auto kv : prev_arr) {
                        GradCell u = kv.first;
                        Tensor *v = kv.second;
                        prev_arr_map[u.dst].push_back({u.src, v});
                    }

                    for (auto kv : curr_grad_map) {
                        GradCell u = kv.first;
                        Tensor *v = kv.second;

                        if (prev_arr_map.count(u.src) > 0) {
                            for (auto z : prev_arr_map[u.src]) {
                                GradCell u_new = {z.node, u.dst};
                                if (new_arr.count(u_new) == 0) new_arr[u_new] = dotp(z.value, v, false);
                                else new_arr[u_new] = add(new_arr[u_new], dotp(z.value, v, false));
                            }
                        }

                        delete [] v->shape;
                        delete [] v->values;
                        delete v;
                    }

                    for (auto kv : prev_arr) {
                        Tensor *v = kv.second;

                        delete [] v->shape;
                        delete [] v->values;
                        delete v;
                    }

                    prev_arr = new_arr;
                }
                else prev_arr = curr_grad_map;

                for (auto kv : prev_arr) {
                    GradCell u = kv.first;
                    Tensor *v = kv.second;
                    NodeFunc<double> *node = u.dst;

                    if (node->num_children == 2 && node->children[1]->is_param) {
                        GradCell z = {u.src, node->children[1]};

                        Tensor *a = node->children[0]->func();
                        Tensor *b = node->children[1]->func();
                        unsigned long b_dim = b->shape[0]*b->shape[1];

                        Tensor *c;

                        if (grad_acc.count(z) == 0) {
                            c = new Tensor();
                            c->n_dim = 1;
                            c->shape = new unsigned int[1];
                            c->shape[0] = b_dim;
                            c->values = new double[b_dim];
                            for (auto j = 0; j < b_dim; j++) c->values[j] = 0.0;
                            grad_acc[z] = c;
                        }
                        
                        c = grad_acc[z];

                        omp_set_num_threads(8);
                        #pragma omp parallel for shared(a, b, c, v)
                        for (auto i1 = 0; i1 < a->shape[0]; i1++) {
                            for (auto j = 0; j < a->shape[1]; j++) {
                                for (auto k = 0; k < b->shape[1]; k++) {
                                    double uw = a->values[i1*a->shape[1]+j]*v->values[i1*v->shape[1]+k];
                                    c->values[j*b->shape[1]+k] += uw;
                                }
                            }
                        }
                    }
                }
            }
        }
};

void generate_regeression_data(double *x, double *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);
        for (auto j = 0; j < m_y; j++) y[i*m_y+j] = dist(engine);
    }
}

void generate_binary_classification_data(double *x, double *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);
        for (auto j = 0; j < m_y; j++) y[i*m_y+j] =(1.0/(1.0+exp(-dist(engine))) > 0.5)?1.0:0.0;
    }
}

void generate_categorical_classification_data(double *x, double *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);

        double s = -__DBL_MAX__;
        unsigned int h = 0;
        for (auto k = 0; k < m_x; k++) {
            double z = exp(-sin(x[i*m_x+k]));
            if (z > s) {
                s = z;
                h = k;
            }
        }

        for (auto j = 0; j < m_y; j++) y[i*m_y+j] = 0.0;
        y[i*m_y+(h % m_y)] = 1.0;
    }
}

Graph *model(unsigned long m_x, unsigned long m_y) {
    Graph *g = new Graph();

    std::random_device rd;
    std::mt19937 engine(rd());

    int m = 3;
    unsigned int *layers = new unsigned int[m];

    layers[0] = m_x;
    layers[1] = 32;
    layers[2] = m_y;

    for (auto i = 0; i < m; i++) {
        if (i == 0) {
            unsigned int *inp_shape = new unsigned int[2];
            unsigned int inp_dim = 2;
            g->root_node = g->_input(inp_shape, inp_dim);
        }
        else {
            std::normal_distribution<double> dist(0.0, sqrt(2.0/layers[i-1]));

            unsigned long h = layers[i-1]*layers[i];
            double *init_v = new double[h];
            for (auto j = 0; j < h; j++) init_v[j] = dist(engine);

            unsigned int *param_shape = new unsigned int[2];
            unsigned int param_dim = 2;
            param_shape[0] = layers[i-1];
            param_shape[1] = layers[i];

            NodeFunc<double> *param_node = g->_parameter(init_v, param_shape, param_dim);
            g->root_node = g->_dot(g->root_node, param_node);

            if (i < m-1) g->root_node = g->_relu(g->root_node);
        }
    }

    g->root_node = g->_sigmoid(g->root_node);

    unsigned int *out_shape = new unsigned int[2];
    unsigned int out_dim = 2;
    NodeFunc<double> *out_node = g->_output(out_shape, out_dim);

    g->root_node = g->_logistic_loss(g->root_node, out_node);
    g->build_dag();

    delete [] layers;

    return g;
}

void fit(double *x, double *y, unsigned long n, unsigned long m_x, unsigned long m_y, unsigned long batch_size, unsigned long n_epochs, double lr, Graph *g) {
    int e = 1;
    while (e <= n_epochs) {
        std::cout << e << std::endl;
        unsigned int j = 0; 
        double loss = 0.0;

        while (j < n) {
            unsigned int new_batch = min(batch_size, n-j);

            unsigned int b = 16;
            unsigned int i = 0;
            while (i < new_batch) {
                unsigned int b2 = min(b, new_batch-i);

                Tensor *x_inp = new Tensor();
                x_inp->n_dim = 2;
                x_inp->shape = new unsigned int[2];
                x_inp->shape[0] = b2;
                x_inp->shape[1] = m_x;
                x_inp->values = x+(j+i)*m_x;

                Tensor *y_oup = new Tensor();
                y_oup->n_dim = 2;
                y_oup->shape = new unsigned int[2];
                y_oup->shape[0] = b2;
                y_oup->shape[1] = m_y;
                y_oup->values = y+(j+i)*m_y;

                g->forward_pass(x_inp, y_oup);
                loss += g->root_node->func()->values[0];

                g->gradient_accumulation();
                g->reset_caches();

                delete x_inp;
                delete y_oup;

                i += b;
            }

            for (auto kv : g->grad_acc) {
                GradCell u = kv.first;
                Tensor *v = kv.second;

                unsigned int m1 = 1;
                for (auto k = 1; k < v->n_dim; k++) m1 *= v->shape[k];
                for (auto k = 0; k < m1; k++) u.dst->param_val->values[k] -= lr*v->values[k]/new_batch;

                delete [] v->shape;
                delete [] v->values;
                delete v;
            }

            g->grad_acc.clear();
            j += batch_size;
        }
        
        std::cout << loss/n << std::endl;
        e++;
    }
}


int main(int argc, char *argv[]) {
    unsigned long n = 100000;
    unsigned long m_x = 128;
    unsigned long m_y = 1;
    unsigned long batch_size = 64;
    unsigned long n_epochs = 100;

    double *x = new double[n*m_x];
    double *y = new double[n*m_y];

    generate_binary_classification_data(x, y, n, m_x, m_y);
    Graph *g = model(m_x, m_y);
    fit(x, y, n, m_x, m_y, batch_size, n_epochs, 0.001, g);

    for (auto i = 0; i < g->dag.size(); i++) {
        LevelData *d = g->dag[i];
        for (auto j = 0; j < d->curr_level_nodes.size(); j++) {
            NodeFunc<double> *node = d->curr_level_nodes[j];
            delete node;
        }
        delete d;
    }

    delete [] x;
    delete [] y;
    delete g;
}