import LogisticRegression
import numpy as np
import random
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n, m = 10000, 1000
x, y = [], []

if rank == 0:
    d = np.random.uniform(0.0, 1.0, (n, m)).tolist()
    x = [x for xs in d for x in xs]
    y = [random.randint(0, 1) for _ in range(n)]
    LogisticRegression.py_lr_train_root(x, y, n, m, size, 0.001, 500, 64, 0.0, 0.0, "lr_model_py", comm)
else:
    LogisticRegression.py_lr_train(n, m, rank, size, 0.001, 500, 64, 0.0, 0.0, "lr_model_py", comm)
