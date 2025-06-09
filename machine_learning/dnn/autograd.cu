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
#include <assert.h>
#include <initializer_list>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define EPSILON 1e-15
#define TILE_WIDTH 16

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
        Tensor* node_val = nullptr;
        unsigned int *oup_shape;

        std::function<Tensor*()> func;
        std::function<Tensor**(Tensor *)> d_func;

        NodeFunc** children;
        unsigned int num_children = 0;

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

unsigned long get_tensor_n_elements(const Tensor *a) {
    unsigned long n = 1;
    for (auto i = 0; i < a->n_dim; i++) n *= a->shape[i];
    return n;
}

bool is_same_shape(const Tensor *a, const Tensor *b) {
    if (a->n_dim == b->n_dim) {
        for (auto i = 0; i < a->n_dim; i++) {
            if (a->shape[i] != b->shape[i]) return false;
        }
        return true;
    }
    return false;
}

Tensor *add(const Tensor *a, const Tensor *b) {
    Tensor *out = new Tensor();

    assert(is_same_shape(a, b));

    out->n_dim = a->n_dim;
    out->shape = new unsigned int[out->n_dim];
    std::copy(a->shape, a->shape+a->n_dim, out->shape);

    unsigned long n = get_tensor_n_elements(a);
    cudaMallocManaged(&out->values, sizeof(float)*n);

    omp_set_num_threads(4);
    #pragma omp parallel for shared(a, b, out)
    for (auto i = 0; i < n; i++) out->values[i] = a->values[i] + b->values[i];

    return out;
}

__global__ 
void cuda_mul(float *a, float *b, float *c, int n, int m, int p) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        float res = 0.0;
        for (int i = 0; i < m; i++) res += a[row*m+i]*b[i*p+col];
        c[row*p+col] = res;
    }
}

__global__ 
void cuda_mul_tiled(float *a, float *b, float *c, int n, int m, int p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    float res = 0.0;
    for (int ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0;


        if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[ty*TILE_WIDTH+tx] = b[(ph*TILE_WIDTH+ty)*p + col];
        else Nds[ty*TILE_WIDTH+tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) res += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
        __syncthreads();
    }

    if (row < n && col < p) c[row*p+col] = res;
}

class Graph {
    private:
    public:
        std::unordered_set<std::string> is_cached;
        std::vector<std::vector<NodeFunc<float>*>> dag;
        std::unordered_map<NodeFunc<float>*, Tensor*> grad_acc;

        unsigned int batch_size = 1;
        NodeFunc<float> *root_node;

        Graph() {}

        NodeFunc<float> *_input(Tensor *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            obj->is_input = true;

            obj->node_val = inp;
            std::string id = obj->id;
            obj->oup_shape = inp->shape;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<float> *_output(Tensor *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            obj->is_output = true;

            obj->node_val = oup;
            std::string id = obj->id;
            obj->oup_shape = oup->shape;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<float> *_parameter(Tensor *param) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            obj->is_param = true;

            obj->node_val = param;
            std::string id = obj->id;
            obj->oup_shape = param->shape;

            obj->func = [obj](){return obj->node_val;};
            obj->d_func = [](Tensor *grad){                
                Tensor **out = new Tensor*[1];
                out[0] = grad; 
                return out;
            };

            return obj;
        }

        NodeFunc<float> *_add(std::vector<NodeFunc<float>*> inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            assert(inp.size() > 0);

            std::string id = obj->id;
            obj->oup_shape = inp.begin()[0]->oup_shape;

            obj->func = [inp, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                
                Tensor *out = new Tensor();

                unsigned int n = inp.begin()[0]->func()->shape[0];
                unsigned int m = inp.begin()[0]->func()->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                cudaMallocManaged(&out->values, sizeof(float)*n*m);
                for (auto i = 0; i < n*m; i++) out->values[i] = 0.0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];

                    assert(n == out->shape[0] && m == out->shape[1]);

                    for (auto i = 0; i < n*m; i++) out->values[i] += a->values[i];
                }

                obj->node_val = out;
                return out;
            };

            obj->d_func = [inp](Tensor *grad){
                Tensor **out = new Tensor*[1];
                
                unsigned int k = 0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];

                    if (grad != nullptr) assert(n == grad->shape[0] && m == grad->shape[1]);

                    out[k] = new Tensor();
                    out[k]->n_dim = 2;
                    out[k]->shape = new unsigned int[2];

                    out[k]->shape[0] = n;
                    out[k]->shape[1] = m;

                    cudaMallocManaged(&out[k]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[k]->values[i] = 0.0;
                    
                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(out, grad)
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < m; j++) {
                            if (grad == nullptr) out[k]->values[i*m+j] += 1.0;
                            else out[k]->values[i*m+j] += grad->values[i*m+j];
                        }
                    }

                    k++;
                }

                return out;
            };

            obj->children = new NodeFunc<float>*[inp.size()];
            unsigned int k = 0;
            for (NodeFunc<float> *x : inp) obj->children[k++] = x;
            obj->num_children = inp.size();

            return obj;
        }

        NodeFunc<float> *_concat(std::vector<NodeFunc<float>*> inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            assert(inp.size() > 0);

            std::string id = obj->id;

            unsigned int p = 0;
            for (NodeFunc<float> *x : inp) p += x->func()->shape[1];

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp.begin()[0]->oup_shape[0];
            obj->oup_shape[1] = p;

            obj->func = [inp, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                
                Tensor *out = new Tensor();

                unsigned int n = inp.begin()[0]->func()->shape[0];
                unsigned int m = inp.begin()[0]->func()->shape[1];

                unsigned int p = 0;
                for (NodeFunc<float> *x : inp) p += x->func()->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = p;

                cudaMallocManaged(&out->values, sizeof(float)*n*p);
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

                obj->node_val = out;
                return out;
            };

            obj->d_func = [inp](Tensor *grad) {
                Tensor **out = new Tensor*[1];

                unsigned int p = 0;
                for (NodeFunc<float> *x : inp) p += x->func()->shape[1];

                if (grad != nullptr) assert(p == grad->shape[1]);

                unsigned int k = 0;
                unsigned int q = 0;

                for (NodeFunc<float> *x : inp) {
                    Tensor *a = x->func();
                    if (grad != nullptr) assert(a->shape[0] == grad->shape[0]);

                    unsigned int n = a->shape[0];
                    unsigned int m = a->shape[1];
                    unsigned int p = grad->shape[1];

                    out[k] = new Tensor();
                    out[k]->n_dim = 2;
                    out[k]->shape = new unsigned int[2];

                    out[k]->shape[0] = n;
                    out[k]->shape[1] = m;

                    cudaMallocManaged(&out[k]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[k]->values[i] = 0.0;
                    
                    for (auto i = 0; i < n; i++) {
                        for (auto j = 0; j < m; j++) {
                            if (grad == nullptr) out[k]->values[i*m+j] += 1.0;
                            else out[k]->values[i*m+j] += grad->values[i*p+j+q];
                        }
                    }

                    k++;
                    q += m;
                }


                return out;
            };

            obj->children = new NodeFunc<float>*[inp.size()];
            unsigned int k = 0;
            for (NodeFunc<float> *x : inp) obj->children[k++] = x;
            obj->num_children = inp.size();

            return obj;
        }

        NodeFunc<float> *_dot(NodeFunc<float> *inp1, NodeFunc<float> *inp2) {
            NodeFunc<float> *obj = new NodeFunc<float>();
            assert(inp1->is_param || inp2->is_param);

            std::string id = obj->id;

            obj->oup_shape = new unsigned int[2];
            Tensor *out = new Tensor();

            if (inp2->is_param) {
                obj->oup_shape[0] = inp1->oup_shape[0];
                obj->oup_shape[1] = inp2->oup_shape[1];

                out->n_dim = 2;
                out->shape = obj->oup_shape;
                cudaMallocManaged(&out->values, sizeof(float)*obj->oup_shape[0]*obj->oup_shape[1]);
            }
            else {
                obj->oup_shape[1] = inp1->oup_shape[0];
                obj->oup_shape[0] = inp2->oup_shape[1];

                out->n_dim = 2;
                out->shape = obj->oup_shape;
                cudaMallocManaged(&out->values, sizeof(float)*obj->oup_shape[0]*obj->oup_shape[1]);
            }

            obj->func = [inp1, inp2, this, id, obj, out](){
                if (obj->node_val != nullptr) return obj->node_val;

                Tensor *a = inp1->func();
                Tensor *b = inp2->func();

                assert(a->shape[1] == b->shape[0]);

                if (inp2->is_param) {
                    unsigned int m = out->shape[0]*out->shape[1];
                    for (auto i = 0; i < m; i++) out->values[i] = 0.0;
                    
                    dim3 bd(16, 16, 1);
                    dim3 gd(ceil(out->shape[0]/16.0), ceil(out->shape[1]/16.0), 1);

                    cuda_mul<<<gd, bd>>>(a->values, b->values, out->values, a->shape[0], a->shape[1], b->shape[1]);
                    
                    cudaDeviceSynchronize();
                }
                else {
                    out->shape[0] = b->shape[0];
                    out->shape[1] = a->shape[1];

                    unsigned int m = out->shape[0]*out->shape[1];

                    cudaMallocManaged(&out->values, sizeof(float)*m);
                    for (auto i = 0; i < m; i++) out->values[i] = 0.0;

                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(a, b, out)
                    for (auto i = 0; i < b->shape[0]; i++) {
                        for (auto j = 0; j < a->shape[0]; j++) {
                            for (auto k = 0; k < a->shape[1]; k++) {
                                out->values[i*a->shape[1]+k] += b->values[i*a->shape[0]+j]*a->values[j*a->shape[1]+k];
                            }
                        }
                    }
                }

                obj->node_val = out;
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

                    cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(b, grad, out)
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

                    cudaMallocManaged(&out[1]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(a, b, grad, out)
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

                    cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(a, grad, out)
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

                    cudaMallocManaged(&out[1]->values, sizeof(float)*n*m);
                    for (auto i = 0; i < n*m; i++) out[1]->values[i] = 0.0;

                    omp_set_num_threads(4);
                    #pragma omp parallel for shared(a, b, grad, out)
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

            obj->children = new NodeFunc<float>*[2];
            obj->children[0] = inp1;
            obj->children[1] = inp2;
            obj->num_children = 2;

            return obj;
        }

        NodeFunc<float> *_relu(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;
            obj->oup_shape = inp->oup_shape;

            obj->func = [inp, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                
                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                cudaMallocManaged(&out->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, out)
                for (auto i = 0; i < n*m; i++) out->values[i] = (a->values[i] > 0.0)?a->values[i]:0.0;

                obj->node_val = out;
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

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    if (grad == nullptr) out[0]->values[i] = (a->values[i] > 0.0)?1.0:0.0;
                    else out[0]->values[i] = grad->values[i]*((a->values[i] > 0.0)?1.0:0.0);
                }

                return out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_sigmoid(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;
            obj->oup_shape = inp->oup_shape;

            obj->func = [inp, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;

                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                cudaMallocManaged(&out->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, out)
                for (auto i = 0; i < n*m; i++) out->values[i] = 1.0/(1.0+exp(-a->values[i]));

                obj->node_val = out;
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

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    float w = 1.0/(1.0+exp(-a->values[i]));
                    if (grad == nullptr) out[0]->values[i] = w*(1.0-w);
                    else out[0]->values[i] = grad->values[i]*w*(1.0-w);
                }
                
                return out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_softmax(NodeFunc<float> *inp) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;
            obj->oup_shape = inp->oup_shape;

            obj->func = [inp, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;

                Tensor *out = new Tensor();
                Tensor *a = inp->func();

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

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

                out->n_dim = 2;
                out->shape = new unsigned int[2];

                out->shape[0] = n;
                out->shape[1] = m;

                cudaMallocManaged(&out->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, maxv, sumv, out)
                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) out->values[i*m+j] = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                }

                obj->node_val = out;
                return out;
            };

            obj->d_func = [inp](Tensor *grad){
                Tensor **out = new Tensor*[1];
                Tensor *a = inp->func();

                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && a->shape[1] == grad->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

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

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);
                for (auto i = 0; i < n*m; i++) out[0]->values[i] = 0.0;

                //dL/dxj = dL/dyk*dyk/dxj

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, maxv, sumv, grad, out)
                for (auto i = 0; i < n; i++) {
                    for (auto j = 0; j < m; j++) {
                        float z1 = exp(a->values[i*m+j]-maxv[i])/sumv[i];
                        float q = 0.0?(a->values[i*m+j] == maxv[i]):1.0; 
                        for (auto k = 0; k < m; k++) {
                            float z2 = exp(a->values[i*m+k]-maxv[i])/sumv[i];

                            if (grad == nullptr) out[0]->values[i*m+j] += ((j == k)?z1*(1.0-z1)*q:-z1*z2*q);
                            else out[0]->values[i*m+j] += grad->values[i*m+k]*((j == k)?z1*(1.0-z1)*q:-z1*z2*q);
                        }
                    }
                }

                return out;
            };

            obj->children = new NodeFunc<float>*[1];
            obj->children[0] = inp;
            obj->num_children = 1;

            return obj;
        }

        NodeFunc<float> *_mse_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;

            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;
            
            obj->func = [inp, oup, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                cudaMallocManaged(&out->values, sizeof(float));
                out->values[0] = 0.0;

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, out)
                for (auto i = 0; i < n*m; i++) out->values[0] += 0.5*(a->values[i]-b->values[i])*(a->values[i]-b->values[i]);

                obj->node_val = out;
                return out;
            };

            obj->d_func = [inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(a->shape[0] == b->shape[0] && grad->shape[1] == 1);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[0]->values[i] = (a->values[i]-b->values[i]);
                    else out[0]->values[i] = grad->values[r]*(a->values[i]-b->values[i]);
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                cudaMallocManaged(&out[1]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[1]->values[i] = (-(a->values[i]-b->values[i]));
                    else out[1]->values[i] = grad->values[r]*(-(a->values[i]-b->values[i]));
                }

                return out;
            };

            obj->children = new NodeFunc<float>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<float> *_logistic_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;
            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;

            obj->func = [inp, oup, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                cudaMallocManaged(&out->values, sizeof(float));

                out->values[0] = 0.0;

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, out)
                for (auto j = 0; j < n*m; j++) out->values[0] += -b->values[j]*log(a->values[j]+EPSILON)-(1.0-b->values[j])*log(1.0-a->values[j]+EPSILON);

                obj->node_val = out;
                return out;
            };

            obj->d_func = [this, id, inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(a->shape[0] == b->shape[0] && grad->shape[1] == 1);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[0]->values[i] = (-b->values[i]*pow(a->values[i]+EPSILON, -1.0)+(1.0-b->values[i])*pow(1.0-a->values[i]+EPSILON, -1.0));
                    else out[0]->values[i] = grad->values[r]*(-b->values[i]*pow(a->values[i]+EPSILON, -1.0)+(1.0-b->values[i])*pow(1.0-a->values[i]+EPSILON, -1.0));
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                cudaMallocManaged(&out[1]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[1]->values[i] = (-log(a->values[i]+EPSILON)+log(1-a->values[i]+EPSILON));
                    else out[1]->values[i] = grad->values[r]*(-log(a->values[i]+EPSILON)+log(1-a->values[i]+EPSILON));
                }
                
                return out;
            };

            obj->children = new NodeFunc<float>*[2];
            obj->num_children = 2;
            obj->children[0] = inp;
            obj->children[1] = oup;

            return obj;
        }

        NodeFunc<float> *_cross_entropy_loss(NodeFunc<float> *inp, NodeFunc<float> *oup) {
            NodeFunc<float> *obj = new NodeFunc<float>();

            std::string id = obj->id;
            obj->oup_shape = new unsigned int[2];
            obj->oup_shape[0] = inp->oup_shape[0];
            obj->oup_shape[1] = 1;

            obj->func = [inp, oup, this, id, obj](){
                if (obj->node_val != nullptr) return obj->node_val;
                Tensor *out = new Tensor();

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out->n_dim = 1;
                out->shape = new unsigned int[1];
                out->shape[0] = 1;

                cudaMallocManaged(&out->values, sizeof(float));

                out->values[0] = 0.0;

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, out)
                for (auto j = 0; j < n*m; j++) out->values[0] += -b->values[j]*log(a->values[j]+EPSILON);

                obj->node_val = out;
                return out;
            };

            obj->d_func = [inp, oup](Tensor *grad){
                Tensor **out = new Tensor*[2];

                Tensor *a = inp->func();
                Tensor *b = oup->func();

                assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
                if (grad != nullptr) assert(a->shape[0] == grad->shape[0] && grad->shape[1] == 1);

                out[0] = new Tensor();
                out[0]->n_dim = 2;
                out[0]->shape = new unsigned int[2];

                unsigned int n = a->shape[0];
                unsigned int m = a->shape[1];

                out[0]->shape[0] = n;
                out[0]->shape[1] = m;

                cudaMallocManaged(&out[0]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[0]->values[i] = -b->values[i]*pow(a->values[i]+EPSILON, -1.0);
                    else out[0]->values[i] = -grad->values[r]*b->values[i]*pow(a->values[i]+EPSILON, -1.0);
                }

                out[1] = new Tensor();
                out[1]->n_dim = 2;
                out[1]->shape = new unsigned int[2];

                out[1]->shape[0] = n;
                out[1]->shape[1] = m;

                cudaMallocManaged(&out[1]->values, sizeof(float)*n*m);

                omp_set_num_threads(4);
                #pragma omp parallel for shared(a, b, grad, out)
                for (auto i = 0; i < n*m; i++) {
                    unsigned int r = i/m;
                    if (grad == nullptr) out[1]->values[i] = -log(a->values[i]+EPSILON);
                    else out[1]->values[i] = -grad->values[r]*log(a->values[i]+EPSILON);
                }
                
                return out;
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
                std::vector<NodeFunc<float>*> nodes = dag[i];

                for (auto nd : nodes) {
                    if (nd->is_input) nd->node_val = x;
                    else if (nd->is_output) nd->node_val = y;

                    nd->func();
                }
            }
        }

        void reset_caches() {
            for (int i = dag.size()-1; i >= 0; i--) {
                for (auto nd : dag[i]) {
                    if (!nd->is_param) nd->node_val = nullptr;
                }
            }
        }
        
        void gradient_accumulation() {
            std::unordered_map<NodeFunc<float>*, Tensor*> grad_map;

            for (auto i = 0; i < dag.size(); i++) {
                std::vector<NodeFunc<float>*> nodes = dag[i];
                for (auto nd : nodes) {
                    Tensor **child_grads;

                    if (grad_map.count(nd) == 0) child_grads = nd->d_func(nullptr);
                    else child_grads = nd->d_func(grad_map[nd]);

                    for (auto j = 0; j < nd->num_children; j++) {
                        NodeFunc<float>* child = nd->children[j];

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
                                cudaMallocManaged(&c->values, sizeof(float)*b_dim);
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
                cudaFree(v->values);
                delete v;
            }
        }

        std::function<NodeFunc<float>*()> input_layer(unsigned int units) {
            return [units, this]() {
                Tensor *inp = new Tensor();
                inp->n_dim = 2;
                inp->shape = new unsigned int[2];
                inp->shape[1] = units;
                return _input(inp);
            };
        }

        std::function<NodeFunc<float>*()> output_layer(unsigned int units) {
            return [units, this]() {
                Tensor *oup = new Tensor();
                oup->n_dim = 2;
                oup->shape = new unsigned int[2];
                oup->shape[1] = units;
                return _output(oup);
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
                float *init_v;
                cudaMallocManaged(&init_v, sizeof(float)*h);
                for (auto j = 0; j < h; j++) init_v[j] = dist(engine);

                Tensor *param = new Tensor();
                param->n_dim = 2;
                param->shape = new unsigned int[2];
                param->shape[0] = n;
                param->shape[1] = m;
                param->values = init_v;

                NodeFunc<float> *param_node = _parameter(param);
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

Graph *model(unsigned long m_x, unsigned long m_y) {
    Graph *g = new Graph();

    std::random_device rd;
    std::mt19937 engine(rd());

    int m = 5;
    unsigned int *layers = new unsigned int[m];

    layers[0] = m_x;
    layers[1] = 32;
    layers[2] = 16;
    layers[3] = 8;
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

void fit(float *x, float *y, unsigned long n, unsigned long m_x, unsigned long m_y, unsigned long batch_size, unsigned long n_epochs, float lr, Graph *g) {
    int e = 1;
    while (e <= n_epochs) {
        std::cout << e << std::endl;
        unsigned int j = 0; 
        float loss = 0.0;

        while (j < n) {
            unsigned int new_batch = min(batch_size, n-j);

            unsigned int b = 64;
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
                NodeFunc<float> *u = kv.first;
                Tensor *v = kv.second;

                for (auto k = 0; k < v->shape[0]; k++) u->node_val->values[k] -= lr*v->values[k]/new_batch;

                delete [] v->shape;
                cudaFree(v->values);
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
    unsigned long m_y = 1;
    unsigned long batch_size = 64;
    unsigned long n_epochs = 100;

    float *x, *y;

    cudaMallocManaged(&x, sizeof(float)*n*m_x);
    cudaMallocManaged(&y, sizeof(float)*n*m_y);

    generate_binary_classification_data(x, y, n, m_x, m_y);
    Graph *g = model(m_x, m_y);
    fit(x, y, n, m_x, m_y, batch_size, n_epochs, 0.001, g);

    for (auto i = 0; i < g->dag.size(); i++) {
        for (auto j = 0; j < g->dag[i].size(); j++) {
            NodeFunc<float> *node = g->dag[i][j];
            cudaFree(node->node_val);
            delete node;
        }
    }

    cudaFree(x);
    cudaFree(y);
    delete g;
}