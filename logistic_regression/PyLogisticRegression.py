from LogisticRegression import LogisticRegression
import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_train, X_test, Y_train, Y_test = None, None, None, None

if rank == 0:
    # n, m = 10000, 1000
    # X, Y = make_classification(n_samples=n, n_features=m, random_state=42)
    
    X, Y = load_breast_cancer(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    n_train, m = X_train.shape
    n_test, m = X_test.shape
    dims = (n_train, n_test, m)
else:
    dims = None

dims = comm.bcast(dims, root=0)

n_train, n_test, m = dims

os.makedirs("models", exist_ok=True)
lr = LogisticRegression(0.001, 100000, 512, m, 0.0, 0.025, "models/bcancer_model_py", comm)
    
if rank == 0:    
    x = X_train.reshape((n_train*m,))
    y = Y_train.astype(np.uint32)
    
    lr.fit_root(x, y, n_train)
    
    x = X_test.reshape((n_test*m,))
    y = Y_test.astype(np.uint32)
    
    out = lr.predict_root(x, n_test)
    print(sum([out[i] == y[i] for i in range(n_test)])/n_test)
    
else:
    lr.fit_non_root(n_train)
    lr.predict_non_root(n_test)
