import torch
import matplotlib.pyplot as plt
import sys
if len(sys.argv) != 3:
    print("wrong number of command line argument")
    sys.exit(1)
sentence = sys.argv[1]
window = int(sys.argv[2])
window = window // 2
sentence = sentence.replace(",", " ")
sentence = sentence.replace("!", " ")
sentence = sentence.replace("?", " ")
sentence = sentence.replace(".", " ")
words = sentence.split()
if len(words) == 0:
    print("the input must contains none delimeters")
srt = sorted(words)
dt = []
for i in range(len(srt)):
    if i == 0 or srt[i] != srt[i-1]:
        dt.append(srt[i])
dic = {}
sz = 0
for ss in srt:
    if ss not in dic:
        dic[ss] = sz
        sz += 1

cooccurence = torch.zeros(sz, sz)
for i in range(len(words)):
    id1 = dic[words[i]]
    for j in range(max(0, i - window), min(i + window + 1, len(words))):
        if i == j:
            continue
        id2 = dic[words[j]]
        cooccurence[id1][id2] += 1

print("the word is ordered as", dt)
print("the cooccurence matrix of window", window * 2, "is:")
print(cooccurence)
U, S, Vh = torch.linalg.svd(cooccurence)
print("svd of the cooccurence matrix is:")
print(U)
print(S)
print(Vh)
Up = U[:,:2]
X = []
Y = []
for i in range(len(Up)):
    X.append(Up[i][0])
    Y.append(Up[i][1])
plt.scatter(X, Y)
for i in range(len(Up)):
    plt.annotate(dt[i], (X[i], Y[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()