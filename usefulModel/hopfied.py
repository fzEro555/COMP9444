import torch
import numpy as np
# Q1
X = torch.tensor(np.array([[1,-1, 1, -1, 1, 1], [1,1,1,-1, -1, -1]]))
p = len(X)
d = X.shape[1]
W = torch.zeros(d, d)
for t in range(0, p):
    for i in range(0, d):
        for j in range(0, d): 
            W[i][j] += X[t][i] * X[t][j]
        W[i][i] = 0
W /= d
print(W)

# Q2
print(W.matmul(X[0].float()))

print(W.matmul(X[1].float()))

