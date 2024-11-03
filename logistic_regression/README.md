# hpc_mpi
High Performance Computing with MPI

## Compiling and Testing C++ Library
mpicxx -O3 -o logistic_regression logistic_regression.cpp -mavx512f

mpirun --mca orte_base_help_aggregate 0 -n 4 ./logistic_regression 10000 1000 0.001 100 32 0.0 0.0 "lr_model"

## Compiling and Testing Cython Library
CC=mpicxx CXX=mpicxx CFLAGS="-O3 -mavx512f" python setup.py build_ext --inplace

mpirun -n 4 python3 PyLogisticRegression.py
