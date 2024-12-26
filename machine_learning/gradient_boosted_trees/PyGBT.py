from GBT import GBT
import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_train, X_test, Y_train, Y_test = None, None, None, None

if rank == 0:
    # n, m = 10000, 100
    # X, Y = make_classification(n_samples=n, n_features=m, random_state=42, n_classes=7, shuffle=True, n_informative=80)
    
    X, Y = fetch_covtype(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 20000, random_state = 42)
    
    enc = LabelEncoder()
    Y_train = enc.fit_transform(Y_train)
    Y_test = enc.transform(Y_test)
    
    sc = RobustScaler()
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
v = GBT(m, 7, 20, 10, 2, 1.0, 0.0, 0.3, 1.0, 1.0, "histogram", "models/gbt_model", comm)
    
if rank == 0:    
    x = X_train.reshape((n_train*m,))
    y = Y_train.astype(np.uint32)
    
    v.fit(x, y, n_train)
    # v.save_model()
    
    # v.load_model("models/gbt_model")
    
    x = X_test.reshape((n_test*m,))
    y = Y_test.astype(np.uint32)
    
    out = v.predict(x, n_test)
    print(sum([out[i] == y[i] for i in range(n_test)])/n_test)
    
else:
    v.fit(np.empty(shape=(n_train*m,)), np.empty(shape=(n_train,), dtype=np.uint32), n_train)
    v.predict(np.empty(shape=(n_test*m,)), n_test)