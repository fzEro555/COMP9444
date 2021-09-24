import numpy as np
# sample
X = np.array([[-2, 2], [2, 1], [-1, -1]])
# class
Y = np.array([1, 0, 0]) 
# weights
w = np.array([0.5, 1, -2])
# learning rate
r = 1.0
# preprocess, insert 1 at the front
X = np.insert(X, 0, 1, axis = 1)
len = X.shape[0]
#set the limit of iterations
i, limit = 1, 100
converge = False
while i <= limit:
    converge = True
    print("----------------iteration " + str(i) + " ------------------")
    for j in range(0, len):
        res = 1
        if X[j].dot(w.T) < 0:
            res = 0
        if res != Y[j]:
            w = w + r * (Y[j] - res) * X[j]
            info = "subtract"
            if res == 0 and Y[j] == 1:
                info = "add"    
            print("sample " + str(X[j, 1:]) + " weight changed to " + str(w) + " " + info)
            converge = False
        else:
            print("sample " + str(X[j, 1:]) + " weight unchanged")
    if converge:
        break
    i += 1
if converge == True:
    print("final weight= " + str(w))
else:
    print("fail to converge within " + str(limit) + " iterations")