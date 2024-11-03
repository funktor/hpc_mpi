# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8

cdef extern from *:
    """
    #include <mpi.h>
    
    #if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
    typedef void *PyMPI_MPI_Message;
    #define MPI_Message PyMPI_MPI_Message
    #endif
    
    #if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
    typedef void *PyMPI_MPI_Session;
    #define MPI_Session PyMPI_MPI_Session
    #endif"
    """

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *
from libcpp.string cimport string
from cpython cimport array
cimport numpy as np

cdef extern from "logistic_regression.h":
    cdef cppclass logistic_regression:
        logistic_regression() except +
        logistic_regression(
                double learning_rate, 
                int epochs, 
                int batch_size, 
                int n_features, 
                double l1_reg,
                double l2_reg, 
                string model_path,
                MPI_Comm comm) except +

        void fit_root(double *x, unsigned int *y, int n)
        void fit_non_root(int n)
        unsigned int *predict_root(double *x, int n)
        void predict_non_root(int n)

cdef convert_int_ptr_to_python(unsigned int *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef class LogisticRegression(object):
    cdef logistic_regression lr
    
    def __cinit__(self, double learning_rate, int num_epochs, int batch_size, int num_features, double l1_reg, double l2_reg, string model_path, MPI.Comm comm):
        self.lr = logistic_regression(learning_rate, num_epochs, batch_size, num_features, l1_reg, l2_reg, model_path, comm.ob_mpi)

    def fit_root(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, np.ndarray[np.uint32_t, ndim=1, mode='c'] y, int n):   
        cdef double *x_arr = &x[0]
        cdef unsigned int *y_arr = &y[0]
        self.lr.fit_root(x_arr, y_arr, n)
    
    def fit_non_root(self, int n): 
        self.lr.fit_non_root(n)
    
    def predict_root(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, n):
        cdef double *x_arr = &x[0]
        return convert_int_ptr_to_python(self.lr.predict_root(x_arr, n), n)
        
    def predict_non_root(self, n):
        self.lr.predict_non_root(n)
