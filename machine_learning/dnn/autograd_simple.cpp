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
// #include <omp.h>
#include <assert.h>

using namespace std;

#define EPSILON 1e-15

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
};

template<typename T>
class NodeFunc {
    private:
    public:
        std::string id;
        bool is_param = false;
        bool is_input = false;
        bool is_output = false;
        Tensor* node_val;

        std::function<Tensor*()> func;
        std::function<Tensor**(Tensor *)> d_func;

        NodeFunc** children;
        unsigned long num_children = 0;

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
    return std::hash<std::string>()(k->id);
  }
};

unsigned long get_tensor_n_elements(Tensor *a) {
    unsigned long n = 1;
    for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];
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

void copy_tensor(Tensor *src, Tensor *dst) {
    dst->n_dim = src->n_dim;
    dst->shape = new unsigned int[dst->n_dim];
    std::copy(src->shape, src->shape+src->n_dim, dst->shape);

    unsigned int n = 1;
    for (auto i = 0; i < dst->n_dim; i++) n *= dst->shape[i];

    dst->values = new double[n];
    std::copy(src->values, src->values+n, dst->values);
}

Tensor *add(Tensor *a, Tensor *b) {
    Tensor *out = new Tensor();

    if (is_same_shape(a, b)) {
        out->n_dim = a->n_dim;
        out->shape = new unsigned int[out->n_dim];
        std::copy(a->shape, a->shape+a->n_dim, out->shape);

        unsigned long n = get_tensor_n_elements(a);
        out->values = new double[n];
        for (auto i = 0; i < n; i++) out->values[i] = a->values[i] + b->values[i];

        return out;
    }

    return nullptr;
}

class Graph {
    private:
    public:
        std::unordered_map<std::string, Tensor*> func_cached;
        std::vector<std::vector<NodeFunc<double>*>> dag;
        std::unordered_map<NodeFunc<double>*, Tensor*> grad_acc;

        unsigned int batch_size = 1;
        NodeFunc<double> *root_node;

        Graph() {}

        NodeFunc<double> *_input(Tensor *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_input = true;

            obj->node_val = inp;
            std::string id = obj->id;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<double> *_output(Tensor *oup) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_output = true;

            obj->node_val = oup;
            std::string id = obj->id;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<double> *_parameter(Tensor *param) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            obj->is_param = true;

            obj->node_val = param;
            std::string id = obj->id;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<double> *_dot(NodeFunc<double> *inp1, NodeFunc<double> *inp2) {
            NodeFunc<double> *obj = new NodeFunc<double>();
            assert(inp1->is_param || inp2->is_param);

            std::string id = obj->id;
            obj->func = [inp1, inp2, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                assert(a->shape[1] == b->shape[0]);

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                if (inp2->is_param) {
                    out->shape[0] = a->shape[0];
                    out->shape[1] = b->shape[1];

                    unsigned int m = out->shape[0]*out->shape[1];

                    out->values = new double[m];
                    for (auto i = 0; i < m; i++) out->values[i] = 0.0;

                    for (auto i = 0; i < a->shape[0]; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                out->values[i*b->shape[1]+k] += a->values[i*b->shape[0]+j]*b->values[j*b->shape[1]+k];
                            }
                        }
                    }
                }
                else {
                    out->shape[0] = b->shape[0];
                    out->shape[1] = a->shape[1];

                    unsigned int m = out->shape[0]*out->shape[1];

                    out->values = new double[m];
                    for (auto i = 0; i < m; i++) out->values[i] = 0.0;

                    for (auto i = 0; i < b->shape[0]; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                out->values[i*a->shape[1]+k] += b->values[i*a->shape[0]+j]*a->values[j*a->shape[1]+k];
                            }
                        }
                    }
                }

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp1, inp2](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                if (inp2->is_param) {
                    //128x32, a=128x64 b=64x32 - 128x64
                    //dL/dxj = dL/dy1*wj1+dL/dy2*wj2+...

                    out[0] = new Tensor();
                    out[0]->n_dim = 2;
                    out[0]->shape = new unsigned int[2];

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];

                    out[0]->shape[0] = n;
                    out[0]->shape[1] = m;

                    if (grad != nullptr) assert(b->shape[1] == grad->shape[1]);

                    out[0]->values = new double[n*m];
                    for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                if (grad == nullptr) out[0]->values[i*b->shape[0]+j] += b->values[j*b->shape[1]+k];
                                else out[0]->values[i*b->shape[0]+j] += grad->values[i*b->shape[1]+k]*b->values[j*b->shape[1]+k];
                            }
                        }
                    }

                    //128x32, a=128x64 b=64x32 - 128x64x32
                    //dL/dwjk = dL/dyk*dyk/dwjk

                    out[1] = new Tensor();
                    out[1]->n_dim = 2;
                    out[1]->shape = new unsigned int[2];

                    m = b->shape[0]*b->shape[1];

                    out[1]->shape[0] = n;
                    out[1]->shape[1] = m;

                    out[1]->values = new double[n*m];
                    for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                if (grad == nullptr) out[1]->values[i*m+j*b->shape[1]+k] += a->values[i*b->shape[0]+j];
                                else out[1]->values[i*m+j*b->shape[1]+k] += grad->values[i*b->shape[1]+k]*a->values[i*b->shape[0]+j];
                            }
                        }
                    }

                    return out;
                }

                else {
                    //128x32, a=128x64 b=64x32 - 128x64
                    //dL/dxj = dL/dy1*wj1+dL/dy2*wj2+...

                    out[0] = new Tensor();
                    out[0]->n_dim = 2;
                    out[0]->shape = new unsigned int[2];

                    unsigned int n = b->shape[0];
                    unsigned int m = b->shape[1];

                    out[0]->shape[0] = n;
                    out[0]->shape[1] = m;

                    if (grad != nullptr) assert(a->shape[1] == grad->shape[1]);

                    out[0]->values = new double[n*m];
                    for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                if (grad == nullptr) out[0]->values[i*a->shape[0]+j] += a->values[j*a->shape[1]+k];
                                else out[0]->values[i*a->shape[0]+j] += grad->values[i*a->shape[1]+k]*a->values[j*a->shape[1]+k];
                            }
                        }
                    }

                    //128x32, a=128x64 b=64x32 - 128x64x32
                    //dL/dwjk = dL/dyk*dyk/dwjk

                    out[1] = new Tensor();
                    out[1]->n_dim = 2;
                    out[1]->shape = new unsigned int[2];

                    m = a->shape[0]*a->shape[1];

                    out[1]->shape[0] = n;
                    out[1]->shape[1] = m;

                    out[1]->values = new double[n*m];
                    for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                if (grad == nullptr) out[1]->values[i*m+j*a->shape[1]+k] += b->values[i*a->shape[0]+j];
                                else out[1]->values[i*m+j*a->shape[1]+k] += grad->values[i*a->shape[1]+k]*b->values[i*a->shape[0]+j];
                            }
                        }
                    }

                    return out;
                }
            };

            obj->children = new NodeFunc<double>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }

        NodeFunc<double> *_relu(NodeFunc<double> *inp) {
            NodeFunc<double> *obj = new NodeFunc<double>();

            std::string id = obj->id;
            obj->func = [inp, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                out->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out->values[i] = (a->values[i] > 0.0)?a->values[i]:0.0;

                obj->node_val = out;
                func_cached[id] = out;

                return out;
            };

            obj->d_func = [inp](Tensor *grad){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[0]->values[i*m+j] += ((a->values[i*m+j] > 0.0)?1.0:0.0);
                        else out[0]->values[i*m+j] += grad->values[i*m+j]*((a->values[i*m+j] > 0.0)?1.0:0.0);
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
            obj->func = [inp, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];

                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                out->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out->values[i] = 1.0/(1.0+exp(-a->values[i]));

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp](Tensor *grad){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        double w = 1.0/(1.0+exp(-a->values[i*m+j]));
                        if (grad == nullptr) out[0]->values[i*m+j] += w*(1.0-w);
                        else out[0]->values[i*m+j] += grad->values[i*m+j]*w*(1.0-w);
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
            obj->func = [inp, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];

                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                double *maxv = new double[n];
                double *sumv = new double[n];

                for (auto i = 0; i < n; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < n; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                out->values = new double[n*m];

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) out->values[i*m+j] = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                }

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp](Tensor *grad){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                double *maxv = new double[n];
                double *sumv = new double[n];

                for (auto i = 0; i < n; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < n; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        double z1 = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                        for (auto k = 0; k < m; k++) {
                            double z2 = exp(a->values[i*m+k]-maxv[i])/sumv[i];
                            if (grad == nullptr) out[0]->values[i*m+j] += ((j == k)?z1*(1.0-z1):-z1*z2);
                            else out[0]->values[i*m+j] += grad->values[i*m+j]*((j == k)?z1*(1.0-z1):-z1*z2);
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
            obj->func = [inp, oup, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                out->values = new double[1]; 
                out->values[0] = 0.0;

                for (auto i = 0; i < n*m; i++) out->values[0] += 0.5*(a->values[i]-b->values[i])*(a->values[i]-b->values[i]);

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                if (grad == nullptr) assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                else assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[0]->values[i*m+j] += (a->values[i*m+j]-b->values[i*m+j]);
                        else out[0]->values[i*m+j] += grad->values[i*m+j]*(a->values[i*m+j]-b->values[i*m+j]);
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                out[1]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[1]->values[i*m+j] += (-(a->values[i*m+j]-b->values[i*m+j]));
                        else out[1]->values[i*m+j] += grad->values[i*m+j]*(-(a->values[i*m+j]-b->values[i*m+j]));
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
            obj->func = [inp, oup, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                out->values = new double[1]; 
                out->values[0] = 0.0;

                for (auto j = 0; j < n*m; j++) out->values[0] += -b->values[j]*log(a->values[j]+EPSILON)-(1.0-b->values[j])*log(1.0-a->values[j]+EPSILON);

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [this, id, inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                if (grad == nullptr) assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                else assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[0]->values[i*m+j] += (-b->values[i*m+j]*pow(a->values[i*m+j]+EPSILON, -1.0)+(1.0-b->values[i*m+j])*pow(1.0-a->values[i*m+j]+EPSILON, -1.0));
                        else out[0]->values[i*m+j] += grad->values[i*m+j]*(-b->values[i*m+j]*pow(a->values[i*m+j]+EPSILON, -1.0)+(1.0-b->values[i*m+j])*pow(1.0-a->values[i*m+j]+EPSILON, -1.0));
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                out[1]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[1]->values[i*m+j] += (-log(a->values[i*m+j]+EPSILON)+log(1-a->values[i*m+j]+EPSILON));
                        else out[1]->values[i*m+j] += grad->values[i*m+j]*(-log(a->values[i*m+j]+EPSILON)+log(1-a->values[i*m+j]+EPSILON));
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

            obj->func = [inp, oup, this, id, obj](){
                if (func_cached.count(id) > 0) return func_cached[id];
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                out->values = new double[1];
                out->values[0] = 0.0;

                for (auto j = 0; j < n*m; j++) out->values[0] += -b->values[j]*log(a->values[j]+EPSILON);

                obj->node_val = out;
                func_cached[id] = out;
                return out;
            };

            obj->d_func = [inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                if (grad == nullptr) assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                else assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                out[0]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[0]->values[i*m+j] += -b->values[i*m+j]*pow(a->values[i*m+j]+EPSILON, -1.0);
                        else out[0]->values[i*m+j] += -grad->values[i*m+j]*b->values[i*m+j]*pow(a->values[i*m+j]+EPSILON, -1.0);
                    }
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                out[1]->values = new double[n*m];
                for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        if (grad == nullptr) out[1]->values[i*m+j] += -log(a->values[i*m+j]+EPSILON);
                        else out[1]->values[i*m+j] += -grad->values[i*m+j]*log(a->values[i*m+j]+EPSILON);
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
            };

            std::deque<DQ> dq;
            std::unordered_map<NodeFunc<double>*, unsigned int> node_level;

            dq.push_back({root_node, 0});
            unsigned int max_level = 0;

            while (dq.size() > 0) {
                DQ x = dq.front();
                dq.pop_front();

                NodeFunc<double> *curr_inp = x.node;
                unsigned int level = x.level;
                max_level = max(max_level, level);

                node_level[curr_inp] = level;

                for (auto i = 0; i < curr_inp->num_children; i++) {
                    if (node_level.count(curr_inp->children[i]) == 0 || node_level[curr_inp->children[i]] < level+1) {
                        dq.push_back({curr_inp->children[i], level+1});
                    }
                }
            }

            dag.resize(max_level+1);
            for (auto kv : node_level) dag[kv.second].push_back(kv.first);
        }

        void forward_pass(Tensor *x, Tensor *y) {
            for (int i = dag.size()-1; i >= 0; i--) {
                std::vector<NodeFunc<double>*> nodes = dag[i];

                for (auto nd : nodes) {
                    if (nd->is_input) nd->node_val = x;
                    else if (nd->is_output) nd->node_val = y;

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
            std::unordered_map<NodeFunc<double>*, Tensor*> grad_map;

            for (auto i = 0; i < dag.size(); i++) {
                std::vector<NodeFunc<double>*> nodes = dag[i];
                for (auto nd : nodes) {
                    Tensor **child_grads;

                    if (grad_map.count(nd) == 0) child_grads = nd->d_func(nullptr);
                    else child_grads = nd->d_func(grad_map[nd]);

                    for (auto j = 0; j < nd->num_children; j++) {
                        NodeFunc<double>* child = nd->children[j];

                        if (grad_map.count(child) == 0) grad_map[child] = child_grads[j];
                        else grad_map[child] = add(grad_map[child], child_grads[j]);

                        if (child->is_param) {
                            Tensor *b = grad_map[child];
                            unsigned int b_dim = b->shape[1];

                            Tensor *c;

                            if (grad_acc.count(child) == 0) {
                                c = new Tensor();
                                c->n_dim = 1;
                                c->shape = new unsigned int[1];
                                c->shape[0] = b_dim;
                                c->values = new double[b_dim];
                                for (auto j1 = 0; j1 < b_dim; j1++) c->values[j1] = 0.0;
                                grad_acc[child] = c;
                            }
                            
                            c = grad_acc[child];

                            for (auto i1 = 0; i1 < b->shape[0]; i1++) {
                                for (auto j1 = 0; j1 < b->shape[1]; j1++) {
                                    c->values[j1] += b->values[i1*b->shape[1]+j1];
                                }
                            }
                        }
                    }
                }
            }

            for (auto kv : grad_map) {
                Tensor *v = kv.second;

                delete [] v->shape;
                delete [] v->values;
                delete v;
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
    layers[1] = 128;
    layers[2] = m_y;

    for (auto i = 0; i < m; i++) {
        if (i == 0) {
            Tensor *inp = new Tensor();
            inp->n_dim = 2;
            inp->shape = new unsigned int[2];
            g->root_node = g->_input(inp);
        }
        else {
            std::normal_distribution<double> dist(0.0, sqrt(2.0/layers[i-1]));

            unsigned long h = layers[i-1]*layers[i];
            double *init_v = new double[h];
            for (auto j = 0; j < h; j++) init_v[j] = dist(engine);

            Tensor *param = new Tensor();
            param->n_dim = 2;
            param->shape = new unsigned int[2];
            param->shape[0] = layers[i-1];
            param->shape[1] = layers[i];
            param->values = init_v;

            NodeFunc<double> *param_node = g->_parameter(param);
            g->root_node = g->_dot(g->root_node, param_node);

            if (i < m-1) g->root_node = g->_relu(g->root_node);
        }
    }

    g->root_node = g->_softmax(g->root_node);

    Tensor *oup = new Tensor();
    oup->n_dim = 2;
    oup->shape = new unsigned int[2];
    NodeFunc<double> *out_node = g->_output(oup);

    g->root_node = g->_cross_entropy_loss(g->root_node, out_node);
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
                NodeFunc<double> *u = kv.first;
                Tensor *v = kv.second;

                for (auto k = 0; k < v->shape[0]; k++) u->node_val->values[k] -= lr*v->values[k]/new_batch;

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
    unsigned long n = 10000;
    unsigned long m_x = 128;
    unsigned long m_y = 5;
    unsigned long batch_size = 64;
    unsigned long n_epochs = 100;

    double *x = new double[n*m_x];
    double *y = new double[n*m_y];

    generate_categorical_classification_data(x, y, n, m_x, m_y);
    Graph *g = model(m_x, m_y);
    fit(x, y, n, m_x, m_y, batch_size, n_epochs, 0.001, g);

    for (auto i = 0; i < g->dag.size(); i++) {
        for (auto j = 0; j < g->dag[i].size(); j++) {
            NodeFunc<double> *node = g->dag[i][j];
            delete node;
        }
    }

    delete [] x;
    delete [] y;
    delete g;
}