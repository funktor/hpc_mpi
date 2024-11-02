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

cdef extern from "logistic_regression.h":
    cdef cppclass logistic_regression:
        logistic_regression() except +
        logistic_regression(
                double learning_rate, 
                int num_epochs, 
                int batch_size, 
                int num_features, 
                double l1_reg,
                double l2_reg) except +

        void fit(
                double *x_train, 
                int *y_train, 
                int n)

        int *predict(double *x_test, int n)
        double *predict_proba(double *x_test, int n)
    
    void build_model(double *x, int *y, int n, int n_features, double learning_rate, int epochs, int batch_size, double l1_reg, double l2_reg, string model_path)
    int *predict_model(double *x, int n, int n_features, string model_path)
    void lr_train(int n, int n_features, int rank, int n_process, double learning_rate, int epochs, int batch_size, double l1_reg, double l2_reg, string model_path, MPI_Comm comm)
    void lr_train_root(double *x, int *y, int n, int n_features, int n_process, double learning_rate, int epochs, int batch_size, double l1_reg, double l2_reg, string model_path, MPI_Comm comm)
    void lr_predict(int n, int n_features, int rank, int n_process, string model_path, MPI_Comm comm)
    int *lr_predict_root(double *x, int n, int n_features, int n_process, string model_path, MPI_Comm comm)


cdef convert_double_ptr_to_python(double *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef convert_int_ptr_to_python(int *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef class LogisticRegression(object):
    cdef logistic_regression lr
    
    def __init__(self, learning_rate, num_epochs, batch_size, num_features, l1_reg, l2_reg):
        self.lr = logistic_regression(learning_rate, num_epochs, batch_size, num_features, l1_reg, l2_reg)
            
    def fit(self, x_train, y_train, n):
        cdef array.array x_arr = array.array('d', x_train)
        cdef array.array y_arr = array.array('i', y_train)
        self.lr.fit(x_arr.data.as_doubles, y_arr.data.as_ints, n)
    
    def predict(self, x_test, n):
        cdef array.array x_arr = array.array('d', x_test)
        return convert_int_ptr_to_python(self.lr.predict(x_arr.data.as_doubles, n), n)
        
    def predict_proba(self, x_test, n):
        cdef array.array x_arr = array.array('d', x_test)
        return convert_double_ptr_to_python(self.lr.predict_proba(x_arr.data.as_doubles, n), n)

def py_train(x, y, n, m, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path):
    cdef array.array x_arr = array.array('d', x)
    cdef array.array y_arr = array.array('i', y)
    build_model(x_arr.data.as_doubles, y_arr.data.as_ints, n, m, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path)

def py_predict(x, n, m, model_path):
    cdef array.array x_arr = array.array('d', x)
    return convert_int_ptr_to_python(predict_model(x_arr.data.as_doubles, n, m, model_path), n)

def py_lr_train_root(x, y, n, m, n_process, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path, MPI.Comm comm):
    cdef array.array x_arr = array.array('d', x)
    cdef array.array y_arr = array.array('i', y)
    lr_train_root(x_arr.data.as_doubles, y_arr.data.as_ints, n, m, n_process, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path, comm.ob_mpi)

def py_lr_train(n, m, rank, n_process, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path, MPI.Comm comm):
    lr_train(n, m, rank, n_process, learning_rate, epochs, batch_size, l1_reg, l2_reg, model_path, comm.ob_mpi)

def py_lr_predict_root(x, n, m, n_process, model_path, MPI.Comm comm):
    cdef array.array x_arr = array.array('d', x)
    lr_predict_root(x_arr.data.as_doubles, n, m, n_process, model_path, comm.ob_mpi)

def py_lr_predict(n, m, rank, n_process, model_path, MPI.Comm comm):
    lr_predict(n, m, rank, n_process, model_path, comm.ob_mpi)



