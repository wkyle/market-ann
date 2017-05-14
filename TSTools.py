#!/usr/bin/python3

#import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import filters as filt
from datetime import date
import statsmodels.api as sm
lowess = sm.nonparametric.lowess


#_____FUNCTIONS_____

def derivOne(t, p, order=1):
    '''
    returns 1st order backwards finite difference of 1st or 2nd order accuracy
    '''
    if len(t) != len(p):
        return None
    if order == 1:
        if len(t) < 2:
            return None
        else:
            h = abs(t[-1]-t[-2])
            return (p[-1]-p[-2])/h
    elif order == 2:
        if len(t) < 3:
            return None
        else:
            h = (abs(t[-1]-t[-2]) + abs(t[-2]-t[-3]))/2
            return (3*p[-1] - 4*p[-2] + p[-3])/(2*h)
    else:
        return None


def derivTwo(t, p, order=1):
    '''
    returns 2nd order backwards finite difference of 1st or 2nd order accuracy
    '''
    if len(t) != len(p):
        return None
    if order == 1:
        if len(t) < 3:
            return None
        else:
            h = (abs(t[-1]-t[-2]) + abs(t[-2]-t[-3]))/2
            return (p[-1] - 2*p[-2] + p[-3])/h**2
    elif order == 2:
        if len(t) < 4:
            return None
        else:
            h = (abs(t[-1]-t[-2]) + abs(t[-2]-t[-3]) + abs(t[-3]-t[-4]))/3
            return (2*p[-1] - 5*p[-2] + 4*p[-3] - p[-4])/h**2
    else:
        return None


def convert_date(date):
    '''
    convert dates from d-mmm-yr to yyyy-mm-dd
    '''
    dic = {'jan' : '01',
           'feb' : '02',
           'mar' : '03',
           'apr' : '04',
           'may' : '05',
           'jun' : '06',
           'jul' : '07',
           'aug' : '08',
           'sep' : '09',
           'oct' : '10',
           'nov' : '11',
           'dec' : '12'}
    day, month, year = date.split("-")
    if len(day) == 1:
        day = '0' + day
    month = dic[month.lower()]
    year = '20' + year
    date = year + '-' + month + '-' + day
    return  date


def fix_google_date(filename):
    data = list(np.genfromtxt(filename, dtype='str', comments="#"))
    head = '#Date\tOpen\tHigh\tLow\tClose\tVolume\n'
    data = [[convert_date(i[0]), i[1], i[2], i[3], i[4], i[5]] for i in data]
    with open(filename, 'w') as f:
        f.write(head)
        [f.write('\t'.join(i) + '\n') for i in data]


def get_ts(symbol, frm=0.0, to=1.0):
    filename = '/home/wes/Stock/db10/' + symbol + '-TS.dat'
    data = list(np.genfromtxt(filename, dtype='str', comments="#"))
    start = int(len(data)*frm)
    end = int(len(data)*to)
    t = [float(i[0]) for i in data[start:end]]
    date = [i[1] for i in data[start:end]]
    p = [float(i[2]) for i in data[start:end]]
    return t,p,date



class growthTester():

    def __init__(self, lam, symbol, init_value, q):
        self.ts_file = '/home/wes/Stock/db10/' + symbol + '-TS.dat'
        self.lam = lam
        self.symbol = symbol
        self.q = q
        self.init_value = init_value

    def get_decision_list(self):
        self.data = list(np.genfromtxt(self.ts_file, dtype='str', comments="#"))[-1000:]
        self.t = [float(i[0]) for i in self.data]
        self.dates = [i[1] for i in self.data]
        self.p = [float(i[2]) for i in self.data]
        #self.p_smooth = filt.gaussian_filter(self.p, self.lam)

        self.buy = [False, False, False, False]
        for index in range(len(self.p[:-4])):
            #self.p_smooth = filt.gaussian_filter(self.p[:index+4], self.lam)
            f = np.poly1d(np.polyfit(self.t[index:index+4], self.p[index:index+4], 2))
            self.p_smooth = [f(i) for i in self.t[index:index+4]]
            slope = derivOne(self.t[index:index+4], self.p_smooth, order=2)
            concavity = derivTwo(self.t[index:index+4], self.p_smooth, order=2)
            measure = self.q*slope + (1-self.q)*concavity
            if measure > 0:
                self.buy.append(True)
            else:
                self.buy.append(False)
        return self.t, self.buy


    def get_account_growth(self):
        self.get_decision_list()
        bank = self.init_value
        nshares = 0
        self.worth = []
        for index, value in enumerate(self.t):
            if self.buy[index] and bank > 0:
                nshares = bank/self.p[index]
                bank = 0
            elif not self.buy[index] and nshares > 0:
                bank = nshares*self.p[index]
                nshares = 0
            else:
                pass
            self.worth.append(nshares*self.p[index] + bank)
        return self.t, self.worth


    def get_percent_growth(self):
        self.get_decision_list()
        self.get_account_growth()
        y1 = int(self.dates[0][0:4])
        m1 = int(self.dates[0][5:7])
        d1 = int(self.dates[0][8:])        
        date1 = date(y1, m1, d1)
        y2 = int(self.dates[-1][0:4])
        m2 = int(self.dates[-1][5:7])
        d2 = int(self.dates[-1][8:])
        date2 = date(y2, m2, d2)
        delta = date2 - date1
        days = delta.days
        return 100*(self.worth[-1]-self.worth[0])/self.worth[0]
        

    def get_total_growth(self):
        self.get_decision_list()
        self.get_account_growth()
        return self.worth[-1]

    def get_yearly_percent_growth(self):    
        self.get_decision_list()
        self.get_account_growth()
        y1 = int(self.dates[0][0:4])
        m1 = int(self.dates[0][5:7])
        d1 = int(self.dates[0][8:])        
        date1 = date(y1, m1, d1)
        y2 = int(self.dates[-1][0:4])
        m2 = int(self.dates[-1][5:7])
        d2 = int(self.dates[-1][8:])
        date2 = date(y2, m2, d2)
        delta = date2 - date1
        days = delta.days
        return 36500*(self.worth[-1]-self.worth[0])/self.worth[0]/days



def get_annual(symbol, start=0.0, stop=1.0, kernel=8, q=.7, mkplt=False, pltsave=True):
    t,p,dates = get_ts(symbol,start,stop)
    avgp = sum(p)/len(p)
    minp = min(p)
    maxp = max(p)
    varlen = 7
    d1 = []
    d2 = []
    for index in range(len(t[4:])):
        d1.append(derivOne(t[index:index+4], p[index:index+4], order=2))
        d2.append(derivTwo(t[index:index+4], p[index:index+4], order=2))
    avg = [sum(p[i:i+4])/4+2*d1[i]-d2[i] for i in range(len(t[4:]))]
    #    d3 = []
    #    for i,j in zip(d1,d2):
    #        if i<0 and j<0:
    #            d3.append(-1)
    #        else:
    #            d3.append(q*i*(1-q)*j)

    d3 = [q*i+(1-q)*j for i,j in zip(d1,d2)]
    movavg = [sum(d3[index:index+kernel])/kernel for index in range(len(t[kernel+4:]))]
    var = []
    for i in range(len(t[varlen:])):
        mu = sum(p[i:i+varlen])/varlen
        var.append(sum([(i-mu)**2 for i in p[i:i+varlen]])/varlen)


    #make a set of shaded boundaries showing buy regions
    buy = False
    init = 100
    bank = init
    nshares = 0
    account = bank
    ranges = []
    for index, value in enumerate(movavg):
        if value>0 and buy == False:
            buy = True
            t1 = t[kernel+index+4]
            nshares = bank/p[kernel+index+4]
            bank = 0
            account = bank + nshares*p[kernel+index+4]
        elif value>0 and buy==True:
            account = bank + nshares*p[kernel+index+4]
        elif value<0 and buy == False:
            account = bank + nshares*p[kernel+index+4]
        elif value<0 and buy == True:
            buy = False
            t2 = t[kernel+index+4]
            bank = nshares*p[kernel+index+4]
            account = bank + nshares*p[kernel+index+4]
            ranges.append([t1+1,t2])
    delta = date(int(dates[-1][0:4]),int(dates[-1][5:7]),int(dates[-1][8:])) - \
            date(int(dates[0][0:4]),int(dates[0][5:7]),int(dates[0][8:]))
    ndays = delta.days


    #this looks good if the rule is buy on second positive day, sell on first negative day
    if mkplt == True:
        plt.plot(t,[(i-minp)*max(movavg)/maxp for i in p], label='ts', lw=.5)
        #plt.plot(t[kernel+4:],movavg, label='avg', lw=.2, color='black')
        #plt.plot(t[4:],d2, lw=.2, color='red')
        #plt.plot(t,p, label='ts', c='blue', lw=.5)
        plt.plot(t[varlen:],var, lw=.3, color='red')
        plt.axhline(linestyle='dashed', lw=.3, color='black')
        #plt.plot(t[4:],[(i-minp)*max(movavg)/maxp for i in avg],lw=.3,c='black')
        #for i in ranges:
        #    plt.axvspan(i[0], i[1], facecolor='g', alpha=0.5, lw=0)

        plt.xlim(min(t[4:]),max(t[4:]))
        #plt.ylim(-.1,max(movavg)*.2)
        plt.legend(loc='upper left')
        plt.xticks(t[0::4],dates[0::4],rotation='vertical')
        if pltsave == True:
            plt.savefig(symbol + '-buy_sell_scheme.png',dpi=500)
        else:
            plt.show()
    return 36500*(account-init)/init/ndays


#_______MAIN_________


'''
#names = ['FMC', 'VZ', 'HSE.TO', 'CL', 'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY', 'AIG']
names = ['MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', 
         'DLX', 'GK', 'PACW'] #volatile high beta
qvals = [i/50 for i in range(50)]
kvals = [i for i in range(1,18)]
Q = [qvals]*len(kvals)
K = [[kvals[i]]*len(qvals) for i in range(len(kvals))]


for name in names:
    print(name)
    for st in [i/100 for i in range(96)]:
        ror = [[0]*len(qvals) for i in range(len(kvals))]
        for indexi, vali in enumerate(qvals):
            for indexj, valj in enumerate(kvals):
                ror[indexj][indexi] = get_annual(name, start=st, stop=st+.05, kernel=valj, q=vali, mkplt=False, pltsave=False)
                print(name,vali,valj,ror[indexj][indexi])

        v = np.linspace(-50, 500, 100, endpoint=True)
        plt.contourf(Q,K,ror, v)
        plt.colorbar(ticks=v[0::10])
        plt.title(name + ' Average Annual Rate of Return (%)')
        plt.xlabel('slope:concavity weight (q)')
        plt.ylabel('moving avg kernel (k)')
        plt.savefig('mod2/'+name+'/' + name + str(int(st*100)) +'-ql-mod2.png',dpi=500)
        plt.cla()
        plt.clf()
'''


'''
names = ['MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', 
         'DLX', 'GK', 'PACW'] #volatile high beta

#for name in names:
#    print(name)
#for q in [i/50 for i in range(1,50)]:
print(get_annual(names[0], start=0.75, stop=1, kernel=2, q=.85, mkplt=True, pltsave=False))
'''

