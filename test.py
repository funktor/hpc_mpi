import LogisticRegression
import numpy as np
import random

n, m = 10000, 1000
d = np.random.uniform(0.0, 1.0, (n, m)).tolist()
x = [x for xs in d for x in xs]
y = [random.randint(0, 1) for _ in range(n)]
LogisticRegression.py_run(x, y, n, m, 0.001, 500, 64)