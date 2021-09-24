import sys
import ast
import math
if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("usage python entropy list of probability [probability]")
li = ast.literal_eval(sys.argv[1])
ret = []
H = 0
for v in li:
    ret.append(1 / v)
    H -= v * math.log2(v)
print("entropy= ", H, " bit vector is ", ret)

if len(sys.argv) == 2:
    exit(0)

q = ast.literal_eval(sys.argv[2])

assert(len(q) == len(li))

KL = 0
RKL = 0
for i in range(len(q)):
    KL += li[i] * (math.log2(li[i]) - math.log2(q[i]))
    RKL += q[i] * (-math.log2(li[i]) + math.log2(q[i]))
print("DKL(p||q) = ", KL, " RDKL(q||p) = ", RKL)