from LogisticRegression import LogisticRegression
import numpy as np
import random
from mpi4py import MPI
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# n, m = 10000, 1000
# X, y = make_classification(n_samples=n, n_features=m, random_state=42)

X_train, X_test, Y_train, Y_test = None, None, None, None

if rank == 0:
    X, Y = load_breast_cancer(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    n1, m = X_train.shape
    n2, m = X_test.shape
    dims = (n1, n2, m)
else:
    dims = None

dims = comm.bcast(dims, root=0)

n1, n2, m = dims
lr = LogisticRegression(0.001, 100000, 512, m, 0.0, 0.025, "bcancer_model_py", comm)
    
if rank == 0:    
    x = X_train.reshape((n1*m,))
    y = Y_train.astype(np.uint32)
    
    lr.fit_root(x, y, n1)
    
    x = X_test.reshape((n2*m,))
    y = Y_test.astype(np.uint32)
    
    out = lr.predict_root(x, n2)
    
    print(out)
    print()
    print(y)
    z = y.tolist()
    print(sum([out[i] == z[i] for i in range(n2)])/len(out))
    
else:
    lr.fit_non_root(n1)
    lr.predict_non_root(n2)
