from SupportVectorMachine import SupportVectorMachine
import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

X, Y = load_breast_cancer(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_train, m = X_train.shape
n_test, m = X_test.shape

os.makedirs("models", exist_ok=True)
v = SupportVectorMachine(m, 1000, 1.0, 1, 0.0, "models/bcancer_model_py", "linear")

x = X_train.reshape((n_train*m,))
y = Y_train.astype(np.uint32)

v.fit(x, y, n_train)

x = X_test.reshape((n_test*m,))
y = Y_test.astype(np.uint32)

out = v.predict(x, n_test)
print(sum([out[i] == y[i] for i in range(n_test)])/n_test)