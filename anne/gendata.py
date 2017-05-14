#!/usr/bin/python3

#import packages
from yahoo_finance import Share
#https://pypi.python.org/pypi/yahoo-finance
from datetime import date, timedelta
import inputs as ip

names = ['WMT', 'FMC', 'VZ', 'HSE.TO', 'CL', 'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY', \
         'MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', \
         'DLX', 'GK', 'PACW', 'AMZN', 'PCLN', 'TRIP', 'ALB', 'TWOU', 'AYA', 'TRU']
with open('names.txt', 'r') as f:
    names = [line.rstrip() for line in f]
db_dir = '/home/wes/Stock/analysis/anne/db/'


def mk_db10():
    shares = [Share(i) for i in names[-50:]]
    print('compiled list of SHARE objects...')
    for stock in shares:
        last10years = reversed(stock.get_historical('2006-08-24', '2016-08-24'))
        sym = stock.get_info()['symbol']
        print("got " + sym + "...")
        with open(db_dir + sym + '-TS.dat', 'w') as f:
            f.write("#index\tdate\tclose\topen\thigh\tlow\tvolume\tpercent_change\tdollar_change\tspread\n")
            for index, i in enumerate(last10years):
                try:
                    percentchange = str(ip.inputs.percent_change(None, float(i['Open']), \
                                                                 float(i['Close'])))
                    dollarchange = str(float(i['Close']) - float(i['Open']))
                    spread = str((float(i['High']) - float(i['Low']))*2 / \
                                 (float(i['Open']) + float(i['Close'])))
                    f.write(str(index) + '\t' + \
                            i['Date'] + '\t' + \
                            i['Close'] + '\t' + \
                            i['Open'] + '\t' + \
                            i['High'] + '\t' + \
                            i['Low'] + '\t' + \
                            i['Volume'] + '\t' + \
                            percentchange + '\t' + \
                            dollarchange + '\t' + \
                            spread + '\t' + \
                            '\n')
                except Exception as e:
                    print(sym,index,e)
                    return

mk_db10()
