from GBT import GBT
import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor

X, Y = load_diabetes(return_X_y=True)
# X, Y = make_regression(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)

# sc = RobustScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

n_train, m = X_train.shape
n_test, m = X_test.shape

print(n_train, n_test, m)

v = GBT(m, 100, 3, 2, 0.9, 0.0, 0.01, 0.7, 0.6, "sorting", "models/gbt_model")

x = X_train.reshape((n_train*m,))
y = Y_train

v.fit(x, y, n_train)

reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, Y_train)

x = X_test.reshape((n_test*m,))
y = Y_test

out = v.predict(x, n_test)
out2 = reg.predict(X_test)
print(y)
print()
print(out)
print(out2)
print(np.sum((np.array(y)-np.array(out))**2))
print(np.sum((np.array(y)-np.array(out2))**2))
    
# if rank == 0:    
    
    
#     v.fit(x, y, n_train)
    
#     x = X_test.reshape((n_test*m,))
#     y = Y_test.astype(np.uint32)
    
#     out = v.predict(x, n_test)
#     print(sum([out[i] == y[i] for i in range(n_test)])/n_test)
    

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# X_train, X_test, Y_train, Y_test = None, None, None, None

# if rank == 0:
#     # n, m = 5000, 10000
#     # X, Y = make_classification(n_samples=n, n_features=m, random_state=42)
    
#     X, Y = load_breast_cancer(return_X_y=True)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, stratify=Y, random_state = 17)
    
#     sc = RobustScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
    
#     n_train, m = X_train.shape
#     n_test, m = X_test.shape
#     dims = (n_train, n_test, m)
# else:
#     dims = None

# dims = comm.bcast(dims, root=0)

# n_train, n_test, m = dims

# os.makedirs("models", exist_ok=True)
# v = SupportVectorMachine(m, 500, 1.0, "models/bcancer_model_py", comm)
    
# if rank == 0:    
#     x = X_train.reshape((n_train*m,))
#     y = Y_train.astype(np.uint32)
    
#     v.fit(x, y, n_train)
    
#     x = X_test.reshape((n_test*m,))
#     y = Y_test.astype(np.uint32)
    
#     out = v.predict(x, n_test)
#     print(sum([out[i] == y[i] for i in range(n_test)])/n_test)
    
# else:
#     v.fit(np.empty(shape=(n_train*m,)), np.empty(shape=(n_train,), dtype=np.uint32), n_train)
#     v.predict(np.empty(shape=(n_train*m,)), n_test)





# # n, m = 10000, 1000
# # X, Y = make_classification(n_samples=n, n_features=m, random_state=42)
    
# # X, Y = load_breast_cancer(return_X_y=True)
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, stratify=Y, random_state = 17)

# # sc = RobustScaler()
# # X_train = sc.fit_transform(X_train)
# # X_test = sc.transform(X_test)

# # n_train, m = X_train.shape
# # n_test, m = X_test.shape

# # os.makedirs("models", exist_ok=True)
# # v = SupportVectorMachine(m, 500, 1.0, 1, 0.0, "models/bcancer_model_py", "linear")

# # svc_lin = SVC(kernel = 'linear', random_state = 0)
# # svc_lin.fit(X_train, Y_train)

# # x = X_train.reshape((n_train*m,))
# # y = Y_train.astype(np.uint32)

# # v.fit(x, y, n_train)

# # out2 = svc_lin.predict(X_test)

# # x = X_test.reshape((n_test*m,))
# # y = Y_test.astype(np.uint32)

# # out = v.predict(x, n_test)

# # print(sum([out[i] == y[i] for i in range(n_test)])/n_test)
# # print(sum([out2[i] == y[i] for i in range(n_test)])/n_test)