import sys
import math
li = sys.argv[1:]
for i in range(len(li)):
    li[i] = float(li[i])

sm = 0
for i in range(len(li)):
    sm += math.exp(li[i])

for i in range(len(li)):
    ans = math.exp(li[i]) / sm
    print("softmax probability ", i + 1, " is ", round(ans, 2), " log probability is ", round(math.log(ans), 2))

prob = []
for i in range(len(li)):
    prob.append(math.exp(li[i]) / sm)

print("please input the target class: ", end=" ")
t = int(input()) - 1
for i in range(len(li)):
    print("d(prob(i)) / d(zi) = ", round((i == t) - prob[i], 2))