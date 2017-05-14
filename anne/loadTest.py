#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import os
from matplotlib.mlab import PCA as mlabPCA


with open('names.txt', 'r') as f:
    names = [line.rstrip() for line in f]


#name1 = 'MDP'
#name2 = 'YHOO'
for i in range(-400,0,1):
    print(i)
    data = []
    for name in names:
        x1, x2, x3, x4, x5, x6 = np.genfromtxt('db/'+name+'-TS-full.dat', comments="#", unpack=True, usecols=(7,8,11,12,13,14))
        data.append([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i]])
    data = np.array(data).T
    pca1 = mlabPCA(data.T)

    plt.plot(pca1.Y[:,0],pca1.Y[:,1], 'o', markersize=7, color='blue', alpha=0.5)
    plt.xlim(-30,30)
    plt.ylim(-15,15)
    plt.savefig('stockEvol' + str(100+i).zfill(4) + '.png')
    plt.close()

os.system("convert -delay 15 stockEvol*.png stockEvolMovie.gif")
os.system("rm stockEvol*.png")




'''
x1 = [i*np.cos(j) for i,j in zip(r1,t1)]
y1 = [i*np.sin(j) for i,j in zip(r1,t1)]
x2 = [i*np.cos(j) for i,j in zip(r2,t2)]
y2 = [i*np.sin(j) for i,j in zip(r2,t2)]

#plt.hist(spread,100)
#plt.xlim(0,10)
#plt.plot([i for i in range(len(spread))], spread,linewidth=.2)#, marker=',',s=1.2, c='red')
#plt.show()

#for i in range(500):
i=1
plt.scatter(x1, y1, c='red', marker='o', s=.5, edgecolors='face')
plt.scatter(x2, y2, c='blue', marker='o', s=.5, edgecolors='face')
plt.savefig('stockEvol' + str(i).zfill(4) + '.png')
plt.close()

#os.system("convert -delay 5 stockEvol*.png stockEvolMovie.gif")
#os.system("rm stockEvol*.png")
'''
