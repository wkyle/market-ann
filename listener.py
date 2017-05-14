#!/usr/bin/python3

#import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from yahoo_finance import Share
#https://pypi.python.org/pypi/yahoo-finance
from TSTools import *
from scipy.ndimage import filters as filt
from datetime import date, timedelta





#____PARAMETERS____

db_dir = '/home/wes/Stock/db10/'
names = ['WMT', 'FMC', 'VZ', 'HSE.TO', 'CL', 'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY']
names = ['MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', 
         'DLX', 'GK', 'PACW'] #volatile high beta


#_____FUNCTIONS_____

def get_current_prices():
    stocks = [Share(i) for i in names]
    status = [{'symbol' : i.get_info()['symbol'], 'price' : i.get_price()} for i in stocks]
    prices = [i['price'] for i in status]
    return prices

def get_last_40():
    stocks = [Share(i) for i in names]
    today = str(date.today())
    past = str(date.today() - timedelta(days=40))
    datalist = []
    for stock in stocks:
        last40 = list(reversed(stock.get_historical(past,today)))
        t = list(range(len(last40)))
        p = [float(i['Adj_Close']) for i in last40]
        sym = stock.get_info()['symbol']
        dates = [i['Date'] for i in last40]
        datalist.append([[sym,i,j,k] for i,j,k in zip(t,dates,p)])
    return datalist

def mk_db10():
    shares = [Share(i) for i in names]
    for stock in shares:
        last5 = reversed(stock.get_historical('2006-08-08', '2016-08-13'))
        sym = stock.get_info()['symbol']
        print(sym)
        with open(db_dir + sym + '-TS.dat', 'w') as f:
            f.write("#index\tdate\tclose\n")
            [f.write(str(index) + '\t' + i['Date'] + '\t' + i['Adj_Close'] + '\n') for index, i in enumerate(last5)]


def get_status(symbol):
    pass






#_______MAIN_________

#mk_db10()
#names = ['MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', 
#         'DLX', 'GK', 'PACW'] #volatile high beta

#[print(i) for i in get_current_prices()]
stocks = get_last_40()

ruleset = [[.1,4], [.4,4], [.3,6], [.55,4], [.6,6], [.2,6], [.3,9], [.5,3], [.2,5], 
           [.5,6], [.6,8], [.6,9], [.9,7]] #q and k

#current_prices = get_current_prices()

for indx, stock in enumerate(stocks):
    sym = stock[0][0]
    t = [i[1] for i in stock]
    p = [i[3] for i in stock]
    #p.append(current_prices[indx])
    #t.append(int(len(t)))

    d1 = []
    d2 = []
    for index in range(len(t[4:])):
        d1.append(derivOne(t[index:index+4], p[index:index+4], order=2))
        d2.append(derivTwo(t[index:index+4], p[index:index+4], order=2))
    d3 = [.7*i+(1-.7)*j for i,j in zip(d1,d2)]

    #movavg = sum(d3[-8:])/8
    if d3[-1] > 0:
        print(sym + ': BUY')
    else:
        print(sym + ': SELL')

