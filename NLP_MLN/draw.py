import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

C=0.7
w_arith1=10
attack="0000"
with open("h2_{}_{}_{}.pkl".format(attack,C,w_arith1), "rb") as tf:
    data = pickle.load(tf)

plt.style.use('seaborn-whitegrid')
print(data[:10])
lis=[(z-x,z-min(1,y)) for x,y,z in data]
lis.sort()
x = [i for i in range(len(lis))]
y=[(x[1]+x[0])/2. for x in lis]
dy=[(x[0]-x[1])/2 for x in lis]
print(dy[-1])
"""
g = []
for i in range(len(data)):
    g.append((data[i][1], data[i][0]))
x=[]
y=[]
dy=[]
cnt=0
g.sort()

for i in range(len(g)):
    cnt += 1
    x.append(cnt - 1)
    d1 = float(g[i][1])
    #d1 = max(float(g[i][1]), 0)
    d2 = float(g[i][0])
    y.append((d1 + d2) / 2.)
    dy.append((d2 - d1) / 2.)
    #y.append(d1)
    #dy.append(d2)
"""	
fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title("attack all eps = {}".format(C))
ax1.set_xlabel('Instances')
ax1.set_ylabel('Output Prob')
plt.axhline(y=C, color='r', linestyle='-')
plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.savefig('single_{}_{}_{}.png'.format(attack,C,w_arith1))
plt.show()
