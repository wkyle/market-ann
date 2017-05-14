#!/usr/bin/python3

#import packages
import numpy as np
from TSTools import *
import inputs as ip




names = ['WMT', 'FMC', 'VZ', 'HSE.TO', 'CL', 'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY', \
         'MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', \
         'DLX', 'GK', 'PACW', 'AMZN', 'PCLN', 'TRIP', 'ALB', 'TWOU', 'AYA', 'TRU']
with open('names.txt', 'r') as f:
    names = [line.rstrip() for line in f]
db_dir = '/home/wes/Stock/analysis/anne/db/'
tsJump = 40


def mkSubDB(name):
    rows = populateRows(name)
    for i in range(tsJump,len(rows)-1):
        inp = ip.inputs()
        slope = inp.onebfd([i.closep for i in rows[i-tsJump:i+1]])
        concavity = inp.twobfd([i.closep for i in rows[i-tsJump:i+1]])
        var = inp.var([i.closep for i in rows[i-tsJump:i+1]])
        entropy = inp.entropy([i.closep for i in rows[i-tsJump:i+1]])
        if rows[i-1].volume != 0:
            dVolume = (rows[i].volume - rows[i-1].volume) / rows[i-1].volume
        else:
            dVolume = 0.0
        if rows[i+1].closep > rows[i].closep:
            answer = 1
        else:
            answer = 0
        rows[i].set_var(var)
        rows[i].set_entropy(entropy)
        rows[i].set_concavity(concavity)
        rows[i].set_slope(slope)
        rows[i].set_dVolume(dVolume)
        rows[i].set_answer(answer)
    write2file(name, rows[tsJump:len(rows)-1])


def mkSubDBNorm(name):
    realrows = get_rows(name)
    rows = [row2input_layer(row) for row in realrows]
    norm_file = db_dir + name + '-Norm.dat'


    dVolume = [i[0] for i in rows]
    mean1 = sum(dVolume)/len(dVolume)
    var1 = np.var(dVolume)

    pchange = [i[1] for i in rows]
    mean2 = sum(pchange)/len(pchange)
    var2 = np.var(pchange)

    spread = [i[2] for i in rows]
    mean3 = sum(spread)/len(spread)
    var3 = np.var(spread)

    vari = [i[3] for i in rows]
    mean4 = sum(vari)/len(vari)
    var4 = np.var(vari)

    entropy = [i[4] for i in rows]
    mean5 = sum(entropy)/len(entropy)
    var5 = np.var(entropy)

    slope = [i[5] for i in rows]
    mean6 = sum(slope)/len(slope)
    var6 = np.var(slope)

    concavity = [i[6] for i in rows]
    mean7 = sum(concavity)/len(concavity)
    var7 = np.var(concavity)

    for row in realrows:
        row.dVolume = (row.dVolume - mean1)/np.sqrt(var1)
        row.percentchange = (row.percentchange - mean2)/np.sqrt(var2)
        row.spread = (row.spread - mean3)/np.sqrt(var3)
        row.var = (row.var - mean4)/np.sqrt(var4)
        row.entropy = (row.entropy - mean5)/np.sqrt(var5)
        row.slope = (row.slope - mean6)/np.sqrt(var6)
        row.concavity = (row.concavity - mean7)/np.sqrt(var7)
    write2fileNorm(name, realrows)


def populateRows(stockname):
    rows = []
    with open(db_dir + stockname + '-TS.dat') as f:
        f.readline()
        for line in f:
            rows.append(rowObject(line, stockname))
    return rows


def write2file(name, rows):
    header = "#index\tdate\tclose\topen\thigh\tlow\tvolume\tdVolume" + \
             "\tpercent_change\tdollar_change\tspread\tvar\tentropy\tslope\tconcavity\tanswer\n"
    filename = db_dir + name + '-TS-full.dat'
    with open(filename, 'w') as f:
        f.write(header)
        [f.write(row.get_string()) for row in rows]

def write2fileNorm(name, rows):
    header = "#index\tdate\tclose\topen\thigh\tlow\tvolume\tdVolume" + \
             "\tpercent_change\tdollar_change\tspread\tvar\tentropy\tslope\tconcavity\tanswer\n"
    filename = db_dir + name + '-Norm.dat'
    with open(filename, 'w') as f:
        f.write(header)
        [f.write(row.get_string()) for row in rows]


for name in names[-50:]:
    try:
        mkSubDB(name)
    except:
        print(name)

