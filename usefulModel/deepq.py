li = [[3, 5],[-4, 1]]
low, high = 0,  0.999
print("-------------------------------------")
r = low
while r <= high:
    print("r= ", r)
    tup = []
    print("policy (0, 1) has V(s1) = ", li[0][0] / (1 - r), " V(s2)= ", li[1][1] / (1 - r))
    tup.append([li[0][0] / (1 - r), li[1][1] / (1 - r), 0, 1])
    
    print("policy (1, 1) has V(s1) = ", li[0][1] + r * li[1][1] / (1 - r), " V(s2)= ", li[1][1] / (1 - r))
    tup.append([li[0][1] + r * li[1][1] / (1 - r), li[1][1] / (1 - r), 1, 1])
    
    print("policy (0, 0) has V(s1) = ", li[0][0] / (1 - r), " V(s2)= ", li[1][0] + r * li[0][0] / (1 - r))
    tup.append([li[0][0] / (1 - r), li[1][0] + r * li[0][0] / (1 - r), 0, 0])
    a = li[0][1]
    b = li[1][0]
    print("policy (1, 0) has V(s1) = ", (a + b * r) / (1 - r * r), " V(s2)= ", (b + a * r) / (1 - r * r))
    tup.append([(a + b * r) / (1 - r * r), (b + a * r) / (1 - r * r), 1, 0])
    tup.sort(reverse=True)
    print("optimal policy is P(S1)= ", tup[0][2] + 1, " P(S2)= ", tup[0][3] + 1)
    print("-------------------------------------")
    r += 0.01
