# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from cpython cimport array
cimport numpy as np

cdef extern from "gbt.h":
    cdef cppclass GradientBoostedTrees:
        GradientBoostedTrees() except +
        GradientBoostedTrees(
                int n_features, 
                int max_num_trees,
                int max_depth_per_tree,
                int min_samples_for_split,
                double reg_const,
                double gamma,
                double lr,
                double feature_sample,
                double data_sample,
                string split_selection_algorithm,
                string model_path) except +

        void fit(double *x, double *y, int n)
        double *predict(double *x, int n)

cdef convert_double_ptr_to_python(double *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef class GBT(object):
    cdef GradientBoostedTrees g
    
    def __cinit__(
            self, 
            int n_features, 
            int max_num_trees,
            int max_depth_per_tree,
            int min_samples_for_split,
            double reg_const,
            double gamma,
            double lr,
            double feature_sample,
            double data_sample,
            string split_selection_algorithm,
            string model_path):

        self.g = GradientBoostedTrees(n_features, max_num_trees, max_depth_per_tree, min_samples_for_split, reg_const, gamma, lr, feature_sample, data_sample, split_selection_algorithm, model_path)

    def fit(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, np.ndarray[np.float64_t, ndim=1, mode='c'] y, int n):   
        cdef double *x_arr = &x[0]
        cdef double *y_arr = &y[0]
        self.g.fit(x_arr, y_arr, n)
    
    def predict(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, n):
        cdef double *x_arr = &x[0]
        return convert_double_ptr_to_python(self.g.predict(x_arr, n), n)


