#!/usr/bin/python3

#import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date
from scipy.ndimage import filters as filt
from scipy.fftpack import fft
from TSTools import *

#parameters

lam = 5



#classes/functions

class epoch():

    def __init__(self, day):
        self.date = day
        self.y0 = int(self.date[0:4])
        self.m0 = int(self.date[5:7])
        self.d0 = int(self.date[8:])
        self.date0 = date(self.y0, self.m0, self.d0)

    def get_day(self, day):
        y = int(day[0:4])
        m = int(day[5:7])
        d = int(day[8:])        
        date1 = date(y, m, d)
        delta = date1 - self.date0
        days = delta.days
        return days


class timeSeries():
    
    def __init__(self, filename):
        self.filename = filename
        self.data = list(np.genfromtxt(filename, dtype='str', comments="#"))
        self.data = sorted([list(i) for i in self.data])
        for index, i in enumerate(self.data):
            self.data[index] = [index] + self.data[index]

    def get_time(self):
        return [int(i[0]) for i in self.data]

    def get_price(self):
        return [float(i[5]) for i in self.data]


#load/define data

ts = timeSeries('VZ.ts')
#ts = timeSeries('WMT.ts')
#ts = timeSeries('FMC.ts')
t = ts.get_time()
p = ts.get_price()



#manipulate data





#lets chart profits if I buy when d>0 and sell when d<0
ql = []
bl = []
q = .5
for lam in [i/10 for i in range(1,100)]:
    ps = filt.gaussian_filter(p,lam)
    d1 = []
    d2 = []
    for i in range(4,len(t)):
        d1.append(derivOne(t[:i], ps[:i], order=2))
        d2.append(derivTwo(t[:i], ps[:i], order=2))
    d = []
    for i,j in zip(d1,d2):
        if i<0 and j<0:
            d.append((q*i)+((1-q)*j))
        else:
            d.append((q*i)+((1-q)*j))


    bank = 5000 #1000 dollars to start with
    nshares = 0
    day = 0
    time = [0]
    account = [100]
    for i,j in zip(p[4:],d):
        if j>0 and bank >0:
            nshares = bank/i
            bank = 0
        elif j<0 and nshares>0:
            bank = nshares*i
            nshares = 0
        else:
            pass
        day += 1
        account.append(bank)
        time.append(day)

    ql.append(lam)
    bl.append(bank)


plt.plot(ql,bl)
plt.savefig('FMC-ratio.pdf')

#show data

'''
plt.subplot(2,1,1)
plt.plot(t,p, lw=.2, c='blue')
plt.plot(t,ps,lw=.3, c='red')
#plt.xlim(1600,2600)

plt.subplot(2,1,2)
plt.fill_between(t[4:], 0, d, facecolor='green', lw=0)
#plt.xlim(1600,2600)
'''
#plt.plot(time,account)

#plt.savefig('VZ-chart.pdf', dpi=500)
