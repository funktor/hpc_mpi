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
from libc.stdlib cimport malloc, free
from cpython cimport array
cimport numpy as np

cdef extern from "svm.h":
    cdef cppclass svm:
        svm() except +
        svm(
            int n_features,
            int max_iter, 
            double C, 
            string model_path, 
            MPI_Comm comm) except +

        void fit(double *x, int *y, int n)
        int *predict(double *x, int n)

cdef convert_int_ptr_to_python(int *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        if ptr[i] == -1:
            lst.append(0)
        else:
            lst.append(1)
    return lst

cdef class SupportVectorMachine(object):
    cdef svm v
    
    def __cinit__(
            self, 
            int n_features,
            int max_iter, 
            double C, 
            string model_path, 
            MPI.Comm comm):

        self.v = svm(n_features, max_iter, C, model_path, comm.ob_mpi)

    def fit(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, np.ndarray[np.uint32_t, ndim=1, mode='c'] y, int n):   
        cdef double *x_arr = &x[0]
        cdef unsigned int *y_arr = &y[0]
        cdef int *y_arr_new = <int *>malloc(n * sizeof(int))
        for i in range(n):
            if y_arr[i] == 0:
                y_arr_new[i] = -1
            else:
                y_arr_new[i] = 1
        self.v.fit(x_arr, y_arr_new, n)
        free(y_arr_new)
    
    def predict(self, np.ndarray[np.float64_t, ndim=1, mode='c'] x, n):
        cdef double *x_arr = &x[0]
        return convert_int_ptr_to_python(self.v.predict(x_arr, n), n)


