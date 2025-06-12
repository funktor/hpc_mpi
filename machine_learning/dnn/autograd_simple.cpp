// Mac instructions: brew install llvm, brew install libomp, g++-15 -O3 autograd.cpp -o autograd -fopenmp
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
#include <assert.h>
#include <initializer_list>
#include "/opt/homebrew/opt/libomp/include/omp.h"

using namespace std;

#define EPSILON 1e-15

template<typename T>
class NodeFunc;

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

void print_vector(std::vector<float> x, size_t n) {
    std::cout << "[";
    for (auto i = 0; i < std::min(n, x.size()); i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

void print_vector(float *x, size_t n) {
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
    float *values;
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

        bool cached = false;
        Tensor* node_val = nullptr;
        unsigned int *oup_shape;

        std::function<Tensor*()> func;
        std::function<Tensor**(Tensor *)> d_func;

        NodeFunc** children;
        unsigned int num_children = 0;

        NodeFunc(std::string type){
            id = get_uuid() + "---" + type;
        }

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

Tensor *add(const Tensor *a, const Tensor *b) {
    Tensor *out = new Tensor();

    assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

    out->n_dim = a->n_dim;
    out->shape = new unsigned int[out->n_dim];
    out->shape[0] = a->shape[0];
    out->shape[1] = a->shape[1];

    unsigned int n = out->shape[0]*out->shape[1];
    out->values = new float[n];

    for (auto i = 0; i < n; i++) out->values[i] = a->values[i] + b->values[i];

    return out;
}

class Graph {
    private:
    public:
        std::vector<std::vector<NodeFunc<float>*>> dag;
        std::unordered_map<NodeFunc<float>*, Tensor*> grad_acc;

        unsigned int batch_size = 1;
        NodeFunc<float> *root_node;

        Graph() {}

        NodeFunc<float> *_input(unsigned int units) {
            NodeFunc<float> *obj = new NodeFunc<float>("input");

            obj->is_input = true;

            obj->node_val = new Tensor();
            obj->node_val->n_dim = 2;
            obj->node_val->shape = new unsigned int[2];
            obj->node_val->shape[0] = batch_size;
            obj->node_val->shape[1] = units;
            obj->node_val->values = new float[batch_size*units];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = batch_size;
            obj->oup_shape[1] = units;

            Tensor **d_out = new Tensor*[1];

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [d_out](Tensor *grad){                
                d_out[0] = grad; 
                return d_out;
            };

            return obj;
        }

        NodeFunc<float> *_output(unsigned int units) {
            NodeFunc<float> *obj = new NodeFunc<float>("output");

            obj->is_output = true;

            obj->node_val = new Tensor();
            obj->node_val->n_dim = 2;
            obj->node_val->shape = new unsigned int[2];
            obj->node_val->shape[0] = batch_size;
            obj->node_val->shape[1] = units;
            obj->node_val->values = new float[batch_size*units];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = batch_size;
            obj->oup_shape[1] = units;

            Tensor **d_out = new Tensor*[1];

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [d_out](Tensor *grad){                
                d_out[0] = grad; 
                return d_out;
            };

            return obj;
        }

        NodeFunc<float> *_parameter(unsigned int n, unsigned int m, float *init_v) {
            NodeFunc<float> *obj = new NodeFunc<float>("param");

            obj->is_param = true;
            
            obj->node_val = new Tensor();
            obj->node_val->n_dim = 2;
            obj->node_val->shape = new unsigned int[2];
            obj->node_val->shape[0] = n;
            obj->node_val->shape[1] = m;
            obj->node_val->values = init_v;

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = m;

            Tensor *c = new Tensor();
            c->n_dim = 1;
            c->shape = new unsigned int[1];
            c->shape[0] = n*m;
            c->values = new float[n*m];
            for (auto i = 0; i < n*m; i++) c->values[i] = 0.0;
            grad_acc[obj] = c;

            Tensor **d_out = new Tensor*[1];

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [d_out](Tensor *grad){                
                d_out[0] = grad; 
                return d_out;
            };

            return obj;
        }

        NodeFunc<float> *_add(std::vector<NodeFunc<float>*> inp) {
            NodeFunc<float> *obj = new NodeFunc<float>("add");
            assert(inp.size() > 0);

            unsigned int n = inp.begin()[0]->oup_shape[0];
            unsigned int m = inp.begin()[0]->oup_shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = m;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = m;
            out->values = new float[n*m];

            obj->func = [inp, this, obj, out, n, m](){
                if (obj->cached) return obj->node_val;

                for (auto i = 0; i < n*m; i++) out->values[i] = 0.0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];
                    assert(n == out->shape[0] && m == out->shape[1]);

                    for (auto i = 0; i < n*m; i++) out->values[i] += a->values[i];
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };

            Tensor **d_out = new Tensor*[inp.size()];

            unsigned int k = 0;
            for (NodeFunc<float> *x : inp) {
                unsigned int n = x->oup_shape[0];
                unsigned int m = x->oup_shape[1];

                d_out[k] = new Tensor();
                d_out[k]->n_dim = 2;
                d_out[k]->shape = new unsigned int[2];
                d_out[k]->shape[0] = n;
                d_out[k]->shape[1] = m;
                d_out[k]->values = new float[n*m];

                k++;
            }

            obj->d_func = [inp, d_out](Tensor *grad){                
                unsigned int k = 0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];
                    if (grad != nullptr) assert(n == grad->shape[0] && m == grad->shape[1]);

                    for (auto i = 0; i < n*m; i++) d_out[k]->values[i] = 0.0;
                    
                    for (auto i = 0; i < n*m; i++) {
                        if (grad == nullptr) d_out[k]->values[i] += 1.0;
                        else d_out[k]->values[i] += grad->values[i];
                    }

                    k++;
                }

                return d_out;
            };

            obj->children = new NodeFunc<float>*[inp.size()];
            k = 0;
            for (NodeFunc<float> *x : inp) obj->children[k++] = x;
            obj->num_children = inp.size();

            return obj;
        }

        NodeFunc<float> *_concat(std::vector<NodeFunc<float>*> inp) {
            NodeFunc<float> *obj = new NodeFunc<float>("concat");
            assert(inp.size() > 0);

            unsigned int n = inp.begin()[0]->oup_shape[0];
            unsigned int m = inp.begin()[0]->oup_shape[1];

            unsigned int p = 0;
            for (NodeFunc<float> *x : inp) p += x->oup_shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = p;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = p;
            out->values = new float[n*p];

            obj->func = [inp, this, obj, out, n, m, p](){
                if (obj->cached) return obj->node_val;

                for (auto i = 0; i < n*p; i++) out->values[i] = 0.0;

                unsigned int k = 0;
                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];

                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < m; j++) {
                            out->values[i*p+j+k] = a->values[i*m+j];
                        }
                    }
                    k += m;
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };


            Tensor **d_out = new Tensor*[inp.size()];
            unsigned int k = 0;

            for (NodeFunc<float> *x : inp) {
                unsigned int n = x->oup_shape[0];
                unsigned int m = x->oup_shape[1];

                d_out[k] = new Tensor();
                d_out[k]->n_dim = 2;
                d_out[k]->shape = new unsigned int[2];
                d_out[k]->shape[0] = n;
                d_out[k]->shape[1] = m;
                d_out[k]->values = new float[n*m];

                k++;
            }

            obj->d_func = [inp, d_out](Tensor *grad) {
                unsigned int p = 0;
                for (NodeFunc<float> *x : inp) p += x->oup_shape[1];

                if (grad != nullptr) assert(p == grad->shape[1]);

                unsigned int k = 0;
                unsigned int q = 0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();
                    if (grad != nullptr) assert(a->shape[0] == grad->shape[0]);

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];
                    unsigned int p = grad->shape[1];

                    for (auto i = 0; i < n*m; i++) d_out[k]->values[i] = 0.0;
                    
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < m; j++) {
                            if (grad == nullptr) d_out[k]->values[i*m+j] += 1.0;
                            else d_out[k]->values[i*m+j] += grad->values[i*p+j+q];
                        }
                    }

                    k++;
                    q += m;
                }
                
                return d_out;
            };

            obj->children = new NodeFunc<float>*[inp.size()];
            k = 0;
            for (NodeFunc<float> *x : inp) obj->children[k++] = x;
            obj->num_children = inp.size();

            return obj;
        }

        NodeFunc<float> *_dot(NodeFunc<float> *inp1, NodeFunc<float> *inp2) {
            NodeFunc<float> *obj = new NodeFunc<float>("dot");
            assert(inp1->is_param || inp2->is_param);

            obj->oup_shape = new unsigned int[2];

            if (inp2->is_param) {
                obj->oup_shape[0] = inp1->oup_shape[0];
                obj->oup_shape[1] = inp2->oup_shape[1];
            }
            else {
                obj->oup_shape[0] = inp2->oup_shape[0];
                obj->oup_shape[1] = inp1->oup_shape[1];
            }

            unsigned int n = obj->oup_shape[0];
            unsigned int m = obj->oup_shape[1];

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = m;
            out->values = new float[n*m];

            obj->func = [inp1, inp2, this, obj, out, n, m](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                if (inp2->is_param) {
                    assert(a->shape[1] == b->shape[0]);

                    for (auto i = 0; i < n*m; i++) out->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(a, b, out)
                    for (auto i = 0; i < a->shape[0]; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                out->values[i*b->shape[1]+k] += a->values[i*b->shape[0]+j]*b->values[j*b->shape[1]+k];
                            }
                        }
                    }
                }
                else {
                    assert(b->shape[1] == a->shape[0]);

                    for (auto i = 0; i < n*m; i++) out->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(a, b, out)
                    for (auto i = 0; i < b->shape[0]; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                out->values[i*a->shape[1]+k] += b->values[i*a->shape[0]+j]*a->values[j*a->shape[1]+k];
                            }
                        }
                    }
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };



            Tensor **d_out = new Tensor*[2];

            if (inp2->is_param) {
                unsigned int n = inp1->oup_shape[0];
                unsigned int m = inp1->oup_shape[1];

                d_out[0] = new Tensor();
                d_out[0]->n_dim = 2;
                d_out[0]->shape = new unsigned int[2];
                d_out[0]->shape[0] = n;
                d_out[0]->shape[1] = m;
                d_out[0]->values = new float[n*m];


                m = inp2->oup_shape[0]*inp2->oup_shape[1];

                d_out[1] = new Tensor();
                d_out[1]->n_dim = 2;
                d_out[1]->shape = new unsigned int[2];
                d_out[1]->shape[0] = n;
                d_out[1]->shape[1] = m;
                d_out[1]->values = new float[n*m];
            }
            else {
                unsigned int n = inp2->oup_shape[0];
                unsigned int m = inp2->oup_shape[1];

                d_out[0] = new Tensor();
                d_out[0]->n_dim = 2;
                d_out[0]->shape = new unsigned int[2];
                d_out[0]->shape[0] = n;
                d_out[0]->shape[1] = m;
                d_out[0]->values = new float[n*m];

                m = inp1->oup_shape[0]*inp1->oup_shape[1];

                d_out[1] = new Tensor();
                d_out[1]->n_dim = 2;
                d_out[1]->shape = new unsigned int[2];
                d_out[1]->shape[0] = n;
                d_out[1]->shape[1] = m;
                d_out[1]->values = new float[n*m];
            }

            obj->d_func = [inp1, inp2, d_out](Tensor *grad){
                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                if (inp2->is_param) {
                    //128x32, a=128x64 b=64x32 - 128x64
                    //dL/dxj = dL/dy1*wj1+dL/dy2*wj2+...

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];

                    if (grad != nullptr) assert(b->shape[1] == grad->shape[1]);

                    for (auto i = 0; i < n*m; i++) d_out[0]->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(b, grad, d_out)
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                if (grad == nullptr) d_out[0]->values[i*b->shape[0]+j] += b->values[j*b->shape[1]+k];
                                else d_out[0]->values[i*b->shape[0]+j] += grad->values[i*b->shape[1]+k]*b->values[j*b->shape[1]+k];
                            }
                        }
                    }

                    //128x32, a=128x64 b=64x32 - 128x64x32
                    //dL/dwjk = dL/dyk*dyk/dwjk

                    m = b->shape[0]*b->shape[1];
                    for (auto i = 0; i < n*m; i++) d_out[1]->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(a, b, grad, d_out)
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < b->shape[0]; j++) {
                            for (auto k = 0; k < b->shape[1]; k++) {
                                if (grad == nullptr) d_out[1]->values[i*m+j*b->shape[1]+k] += a->values[i*b->shape[0]+j];
                                else d_out[1]->values[i*m+j*b->shape[1]+k] += grad->values[i*b->shape[1]+k]*a->values[i*b->shape[0]+j];
                            }
                        }
                    }

                    return d_out;
                }

                else {
                    //128x32, a=128x64 b=64x32 - 128x64
                    //dL/dxj = dL/dy1*wj1+dL/dy2*wj2+...

                    unsigned int n = b->shape[0];
                    unsigned int m = b->shape[1];

                    if (grad != nullptr) assert(a->shape[1] == grad->shape[1]);

                    for (auto i = 0; i < n*m; i++) d_out[0]->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(a, grad, d_out)
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                if (grad == nullptr) d_out[0]->values[i*a->shape[0]+j] += a->values[j*a->shape[1]+k];
                                else d_out[0]->values[i*a->shape[0]+j] += grad->values[i*a->shape[1]+k]*a->values[j*a->shape[1]+k];
                            }
                        }
                    }

                    //128x32, a=128x64 b=64x32 - 128x64x32
                    //dL/dwjk = dL/dyk*dyk/dwjk

                    m = a->shape[0]*a->shape[1];
                    for (auto i = 0; i < n*m; i++) d_out[1]->values[i] = 0.0;

                    omp_set_num_threads(8);
                    #pragma omp parallel for shared(a, b, grad, d_out)
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                if (grad == nullptr) d_out[1]->values[i*m+j*a->shape[1]+k] += b->values[i*a->shape[0]+j];
                                else d_out[1]->values[i*m+j*a->shape[1]+k] += grad->values[i*a->shape[1]+k]*b->values[i*a->shape[0]+j];
                            }
                        }
                    }

                    return d_out;
                }
            };

            obj->children = new NodeFunc<float>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }

        NodeFunc<float> *_relu(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>("relu");

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = m;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = m;
            out->values = new float[n*m];

            obj->func = [inp, this, obj, out, n, m](){
                if (obj->cached) return obj->node_val;
                
                Tensor *a = inp->func();
                for (auto i = 0; i < n*m; i++) out->values[i] = (a->values[i] > 0.0)?a->values[i]:0.0;

                obj->cached = true;
                obj->node_val = out;
                return out;
            };



            Tensor **d_out = new Tensor*[1];
            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            obj->d_func = [inp, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                for (auto i = 0; i < n*m; i++) {
                    if (grad == nullptr) d_out[0]->values[i] = (a->values[i] > 0.0)?1.0:0.0;
                    else d_out[0]->values[i] = grad->values[i]*((a->values[i] > 0.0)?1.0:0.0);
                }

                return d_out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_sigmoid(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>("sigmoid");

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = m;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = m;
            out->values = new float[n*m];

            obj->func = [inp, this, obj, out, n, m](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp->func();
                for (auto i = 0; i < n*m; i++) out->values[i] = 1.0/(1.0+exp(-a->values[i]));

                obj->cached = true;
                obj->node_val = out;
                return out;
            };



            Tensor **d_out = new Tensor*[1];
            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            obj->d_func = [inp, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                for (auto i = 0; i < n*m; i++) {
                    float w = 1.0/(1.0+exp(-a->values[i]));
                    if (grad == nullptr) d_out[0]->values[i] = w*(1.0-w);
                    else d_out[0]->values[i] = grad->values[i]*w*(1.0-w);
                }
                
                return d_out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_softmax(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>("softmax");

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = n;
            obj->oup_shape[1] = m;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = n;
            out->shape[1] = m;
            out->values = new float[n*m];

            obj->func = [inp, this, obj, out, n, m](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp->func();

                float *maxv = new float[n];
                float *sumv = new float[n];

                for (auto i = 0; i < n; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < n; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    out->values[i] = exp(a->values[i]-maxv[r])/sumv[r];
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };



            Tensor **d_out = new Tensor*[1];
            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            obj->d_func = [inp, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                float *maxv = new float[n];
                float *sumv = new float[n];

                for (auto i = 0; i < n; i++) {
                    maxv[i] = -__DBL_MAX__;
                    for (auto j = 0; j < m; j++) maxv[i] = max(maxv[i], a->values[i*m+j]);
                }

                for (auto i = 0; i < n; i++) {
                    sumv[i] = 0.0;
                    for (auto j = 0; j < m; j++) sumv[i] += exp(a->values[i*m+j]-maxv[i]);
                }

                for (auto i = 0; i < n*m; i++) d_out[0]->values[i] = 0.0;

                omp_set_num_threads(8);
                #pragma omp parallel for shared(a, maxv, sumv, grad, d_out)
                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        float z1 = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                        float q = 0.0?(a->values[i*m+j] == maxv[i]):1.0; 
                        for (auto k = 0; k < m; k++) {
                            float z2 = exp(a->values[i*m+k]-maxv[i])/sumv[i];

                            if (grad == nullptr) d_out[0]->values[i*m+j] += q*((j == k)?z1*(1.0-z1):-z1*z2);
                            else d_out[0]->values[i*m+j] += q*grad->values[i*m+k]*((j == k)?z1*(1.0-z1):-z1*z2);
                        }
                    }
                }

                return d_out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_mse_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>("mse");

            assert(inp->oup_shape[0] == oup->oup_shape[0] && inp->oup_shape[1] == oup->oup_shape[1]);

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = inp->oup_shape[0];
            out->shape[1] = 1;
            out->values = new float[obj->oup_shape[0]];
            
            obj->func = [inp, oup, this, obj, out](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                for (auto i = 0; i < n; i++) out->values[i] = 0.0;

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    out->values[r] += 0.5*(a->values[i]-b->values[i])*(a->values[i]-b->values[i]);
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };


            Tensor **d_out = new Tensor*[2];

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            d_out[1] = new Tensor();
            d_out[1]->n_dim = 2;
            d_out[1]->shape = new unsigned int[2];
            d_out[1]->shape[0] = n;
            d_out[1]->shape[1] = m;
            d_out[1]->values = new float[n*m];

            obj->d_func = [inp, oup, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(grad->shape[0] == a->shape[0] && grad->shape[1] == 1);

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[0]->values[i] = (a->values[i]-b->values[i]);
                    else d_out[0]->values[i] = grad->values[r]*(a->values[i]-b->values[i]);
                }

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[1]->values[i] = (-(a->values[i]-b->values[i]));
                    else d_out[1]->values[i] = grad->values[r]*(-(a->values[i]-b->values[i]));
                }

                return d_out;
            };

            obj->children = new NodeFunc<float>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<float> *_logistic_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>("logit");

            assert(inp->oup_shape[0] == oup->oup_shape[0] && inp->oup_shape[1] == oup->oup_shape[1]);

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = inp->oup_shape[0];
            out->shape[1] = 1;
            out->values = new float[inp->oup_shape[0]];

            obj->func = [inp, oup, this, obj, out](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                for (auto i = 0; i < n; i++) out->values[i] = 0.0;

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    out->values[r] += -b->values[i]*log(a->values[i]+EPSILON)-(1.0-b->values[i])*log(1.0-a->values[i]+EPSILON);
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };


            Tensor **d_out = new Tensor*[2];

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            d_out[1] = new Tensor();
            d_out[1]->n_dim = 2;
            d_out[1]->shape = new unsigned int[2];
            d_out[1]->shape[0] = n;
            d_out[1]->shape[1] = m;
            d_out[1]->values = new float[n*m];

            obj->d_func = [this, inp, oup, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(grad->shape[0] == a->shape[0] && grad->shape[1] == 1);

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[0]->values[i] = (-b->values[i]*pow(a->values[i]+EPSILON, -1.0)+(1.0-b->values[i])*pow(1.0-a->values[i]+EPSILON, -1.0));
                    else d_out[0]->values[i] = grad->values[r]*(-b->values[i]*pow(a->values[i]+EPSILON, -1.0)+(1.0-b->values[i])*pow(1.0-a->values[i]+EPSILON, -1.0));
                }

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[1]->values[i] = (-log(a->values[i]+EPSILON)+log(1-a->values[i]+EPSILON));
                    else d_out[1]->values[i] = grad->values[r]*(-log(a->values[i]+EPSILON)+log(1-a->values[i]+EPSILON));
                }
                
                return d_out;
            };

            obj->children = new NodeFunc<float>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<float> *_cross_entropy_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>("xent");

            assert(inp->oup_shape[0] == oup->oup_shape[0] && inp->oup_shape[1] == oup->oup_shape[1]);

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;

            Tensor *out = new Tensor();
            out->n_dim = 2;
            out->shape = new unsigned int[2];
            out->shape[0] = inp->oup_shape[0];
            out->shape[1] = 1;
            out->values = new float[inp->oup_shape[0]];

            obj->func = [inp, oup, this, obj, out](){
                if (obj->cached) return obj->node_val;

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                for (auto i = 0; i < n; i++) out->values[i] = 0.0;

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    out->values[r] += -b->values[i]*log(a->values[i]+EPSILON);
                }

                obj->cached = true;
                obj->node_val = out;
                return out;
            };


            Tensor **d_out = new Tensor*[2];

            unsigned int n = inp->oup_shape[0];
            unsigned int m = inp->oup_shape[1];

            d_out[0] = new Tensor();
            d_out[0]->n_dim = 2;
            d_out[0]->shape = new unsigned int[2];
            d_out[0]->shape[0] = n;
            d_out[0]->shape[1] = m;
            d_out[0]->values = new float[n*m];

            d_out[1] = new Tensor();
            d_out[1]->n_dim = 2;
            d_out[1]->shape = new unsigned int[2];
            d_out[1]->shape[0] = n;
            d_out[1]->shape[1] = m;
            d_out[1]->values = new float[n*m];

            obj->d_func = [inp, oup, d_out, n, m](Tensor *grad){
                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(grad->shape[0] == a->shape[0] && grad->shape[1] == 1);

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[0]->values[i] = -b->values[i]*pow(a->values[i]+EPSILON, -1.0);
                    else d_out[0]->values[i] = -grad->values[r]*b->values[i]*pow(a->values[i]+EPSILON, -1.0);
                }

                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) d_out[1]->values[i] = -log(a->values[i]+EPSILON);
                    else d_out[1]->values[i] = -grad->values[r]*log(a->values[i]+EPSILON);
                }
                
                return d_out;
            };

            obj->children = new NodeFunc<float>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        void build_dag() {
            struct DQ {
                NodeFunc<float> *node;
                unsigned int level;
            };

            std::deque<DQ> dq;
            std::unordered_map<NodeFunc<float>*, unsigned int> node_level;

            dq.push_back({root_node, 0});
            unsigned int max_level = 0;

            while (dq.size() > 0) {
                DQ x = dq.front();
                dq.pop_front();

                NodeFunc<float> *curr_inp = x.node;
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
                for (auto nd : dag[i]) {
                    if (nd->is_input) nd->node_val = x;
                    else if (nd->is_output) nd->node_val = y;

                    nd->func();
                }
            }
        }

        void reset_caches() {
            for (int i = dag.size()-1; i >= 0; i--) {
                for (auto nd : dag[i]) nd->cached = false;
            }
        }
        
        void gradient_accumulation() {
            std::unordered_map<NodeFunc<float>*, Tensor*> grad_map;

            for (auto i = 0; i < dag.size(); i++) {
                for (auto nd : dag[i]) {
                    Tensor **child_grads;

                    if (grad_map.count(nd) == 0) child_grads = nd->d_func(nullptr);
                    else child_grads = nd->d_func(grad_map[nd]);

                    for (auto j = 0; j < nd->num_children; j++) {
                        NodeFunc<float>* child = nd->children[j];
                        if (grad_map.count(child) == 0) grad_map[child] = child_grads[j];
                        else grad_map[child] = add(grad_map[child], child_grads[j]);
                    }
                }
            }

            for (auto kv : grad_map) {
                NodeFunc<float>* child = kv.first;
                if (child->is_param) {
                    Tensor *b = kv.second;
                    Tensor *c = grad_acc[child];

                    for (auto i = 0; i < b->shape[0]; i++) {
                        for (auto j = 0; j < b->shape[1]; j++) {
                            c->values[j] += b->values[i*b->shape[1]+j];
                        }
                    }
                }
            }
        }

        std::function<NodeFunc<float>*()> input_layer(unsigned int units) {
            return [units, this]() {
                return _input(units);
            };
        }

        std::function<NodeFunc<float>*()> output_layer(unsigned int units) {
            return [units, this]() {
                return _output(units);
            };
        }

        std::function<NodeFunc<float>*(NodeFunc<float> *)> dense_layer(unsigned int units, std::string activation) {
            return [units, activation, this](NodeFunc<float> *inp) {
                unsigned int n = inp->oup_shape[1];
                unsigned int m = units;

                static std::random_device rd;
                static std::mt19937 engine(rd());

                std::normal_distribution<float> dist(0.0, sqrt(2.0/n));

                unsigned int h = n*m;
                float *init_v = new float[h];
                for (auto j = 0; j < h; j++) init_v[j] = dist(engine);

                NodeFunc<float> *param_node = _parameter(n, m, init_v);
                NodeFunc<float> *node = _dot(inp, param_node);

                if (activation == "relu") return _relu(node);
                else if (activation == "sigmoid") return _sigmoid(node);
                else if (activation == "softmax") return _softmax(node);
                else return node;
            };
        }
};

void generate_regeression_data(float *x, float *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);
        for (auto j = 0; j < m_y; j++) y[i*m_y+j] = dist(engine);
    }
}

void generate_binary_classification_data(float *x, float *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);
        for (auto j = 0; j < m_y; j++) y[i*m_y+j] =(1.0/(1.0+exp(-dist(engine))) > 0.5)?1.0:0.0;
    }
}

void generate_categorical_classification_data(float *x, float *y, unsigned int n, unsigned int m_x, unsigned int m_y) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < m_x; j++) x[i*m_x+j] = dist(engine);

        float s = -__DBL_MAX__;
        unsigned int h = 0;
        for (auto k = 0; k < m_x; k++) {
            float z = exp(-sin(x[i*m_x+k]));
            if (z > s) {
                s = z;
                h = k;
            }
        }

        for (auto j = 0; j < m_y; j++) y[i*m_y+j] = 0.0;
        y[i*m_y+(h % m_y)] = 1.0;
    }
}

Graph *model(unsigned int m_x, unsigned int m_y, unsigned int batch_size) {
    Graph *g = new Graph();
    g->batch_size = batch_size;

    std::random_device rd;
    std::mt19937 engine(rd());

    int m = 5;
    unsigned int *layers = new unsigned int[m];

    layers[0] = m_x;
    layers[1] = 128;
    layers[2] = 64;
    layers[3] = 32;
    layers[4] = m_y;

    NodeFunc<float>* inp_layer = g->input_layer(layers[0])();
    NodeFunc<float>* layer1    = g->dense_layer(layers[1], "relu")(inp_layer);
    NodeFunc<float>* layer2_1  = g->dense_layer(layers[2], "relu")(layer1);
    NodeFunc<float>* layer2_2  = g->dense_layer(layers[2], "relu")(inp_layer);
    NodeFunc<float>* layer2    = g->_add({layer2_1, layer2_2});
    NodeFunc<float>* layer3_1  = g->dense_layer(layers[3], "relu")(layer2);
    NodeFunc<float>* layer3_2  = g->dense_layer(layers[3], "relu")(layer1);
    NodeFunc<float>* layer3_3  = g->dense_layer(layers[3], "relu")(inp_layer);
    NodeFunc<float>* layer3    = g->_add({layer3_1, layer3_2, layer3_3});
    NodeFunc<float>* layer4    = g->dense_layer(layers[4], "sigmoid")(layer3);

    g->root_node = g->_logistic_loss(layer4, g->output_layer(layers[4])());
    g->build_dag();

    delete [] layers;

    return g;
}

void fit(float *x, float *y, unsigned int n, unsigned int m_x, unsigned int m_y, unsigned int batch_size, unsigned int n_epochs, float lr, Graph *g) {
    int e = 1;
    while (e <= n_epochs) {
        std::cout << e << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        unsigned int j = 0; 
        float loss = 0.0;

        while (j < n) {
            unsigned int new_batch = min(batch_size, n-j);

            unsigned int b = 256;
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
                for (auto k = 0; k < b2; k++) loss += g->root_node->func()->values[k];

                g->gradient_accumulation();
                g->reset_caches();

                delete x_inp;
                delete y_oup;

                i += b;
            }

            for (auto kv : g->grad_acc) {
                NodeFunc<float> *u = kv.first;
                Tensor *v = kv.second;

                for (auto k = 0; k < v->shape[0]; k++) u->node_val->values[k] -= lr*v->values[k]/new_batch;
                for (auto k = 0; k < v->shape[0]; k++) v->values[k] = 0.0;
            }

            j += batch_size;
        }
        
        std::cout << loss/n << std::endl;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken for epoch = " << duration.count() << " milliseconds" << std::endl;

        e++;
    }
}


int main(int argc, char *argv[]) {
    unsigned int n = 100000;
    unsigned int m_x = 128;
    unsigned int m_y = 1;
    unsigned int batch_size = 256;
    unsigned int n_epochs = 100;

    if (n % batch_size != 0) n += (batch_size - (n % batch_size));

    float *x = new float[n*m_x];
    float *y = new float[n*m_y];

    generate_binary_classification_data(x, y, n, m_x, m_y); 
    Graph *g = model(m_x, m_y, batch_size);
    fit(x, y, n, m_x, m_y, batch_size, n_epochs, 0.001, g);

    for (auto kv : g->grad_acc) {
        Tensor *v = kv.second;

        delete [] v->shape;
        delete [] v->values;
        delete v;
    }

    for (auto i = 0; i < g->dag.size(); i++) {
        for (auto j = 0; j < g->dag[i].size(); j++) {
            NodeFunc<float> *node = g->dag[i][j];
            if (!node->is_input && !node->is_output) {
                delete [] node->node_val;
                delete node;
            }
        }
    }

    delete [] x;
    delete [] y;
    delete g;
}
