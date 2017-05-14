#!/usr/bin/python3

#import packages
import numpy as np
from scipy import stats
from numpy.random import uniform
db_dir = '/home/wes/Stock/analysis/anne/db/'




def sigmoid(x):
    beta = 1.0
    return 1 / (1 + np.exp(-beta*x))


def d_sigmoid(x):
    beta = 1.0
    f = sigmoid(x)
    return beta * f * (1 - f)

def tanh(x):
    return np.tanh(1.5*x)

def d_tanh(x):
    #return 1.0 - tanh(x)**2.0
    return 1.5/np.cosh(1.5*x)**2

def relu(x):
    return np.log(1+np.exp(x))

def d_relu(x):
    return np.exp(x)/(np.exp(x)+1)

def gauss(beta, x):
    f = np.exp(-(x/beta)**2)
    return f

def d_gauss(beta, x):
    return -2*x*np.exp(-(x/beta)**2)/beta/beta

class rowObject():

    """
    object to hold rows from raw stock time series.
    """

    def __init__(self, row, symbol):

        """
        row is a string containing a row from the 
        raw time series for a single stock
        """

        lst = row.split()
        self.var = None
        self.entropy = None
        self.slope = None
        self.concavity = None
        self.dVolume = None
        self.answer = None
        self.index = int(lst[0])
        self.date = lst[1]
        self.closep = float(lst[2])
        self.openp = float(lst[3])
        self.high = float(lst[4])
        self.low = float(lst[5])
        self.volume = int(lst[6])
        self.percentchange = float(lst[7])
        self.dollarchange = float(lst[8])
        self.spread = float(lst[9])

        self.symbol = symbol

    def set_var(self, x):
        self.var = x


    def get_var(self):
        return self.var


    def set_entropy(self, x):
        self.entropy = x


    def get_entropy(self):
        return self.entropy


    def set_slope(self, x):
        self.slope = x


    def get_slope(self):
        return self.slope


    def set_concavity(self, x):
        self.concavity = x


    def get_concavity(self):
        return self.concavity


    def set_dVolume(self, x):
        self.dVolume = x


    def get_dVolume(self):
        return self.dVolume


    def set_answer(self, x):
        self.answer = x


    def get_answer(self):
        return self.answer


    def get_string(self):
        if (self.var != None) and (self.entropy != None) and (self.slope != None) and \
           (self.concavity != None) and (self.dVolume != None) and (self.answer != None):
            s = str(self.index) + '\t' + \
                str(self.date) + '\t' + \
                str(self.closep)+ '\t' + \
                str(self.openp) + '\t' + \
                str(self.high) + '\t' + \
                str(self.low) + '\t' + \
                str(self.volume) + '\t' + \
                str(self.dVolume) + '\t' + \
                str(self.percentchange) + '\t' + \
                str(self.dollarchange) + '\t' + \
                str(self.spread) + '\t' + \
                str(self.var) + '\t' + \
                str(self.entropy) + '\t' + \
                str(self.slope) + '\t' + \
                str(self.concavity) + '\t' + \
                str(self.answer) + \
                '\n'
            return s
        else:
            print("Additional info: ", self.var, self.entropy, self.slope, \
                  self.concavity, self.dVolume, self.answer)
            raise Exception("Incomplete rowObject in " + self.symbol + \
                            " - something is missing in this row")



class rowObjectFull():

    """
    object to hold rows from full stock time series.
    """

    def __init__(self, row, symbol):

        """
        row is a string containing a row from the 
        raw time series for a single stock
        """

        lst = row.split()
        self.index = int(lst[0])
        self.date = lst[1]
        self.closep = float(lst[2])
        self.openp = float(lst[3])
        self.high = float(lst[4])
        self.low = float(lst[5])
        self.volume = int(lst[6])
        self.dVolume = float(lst[7])
        self.percentchange = float(lst[8])
        self.dollarchange = float(lst[9])
        self.spread = float(lst[10])
        self.var = float(lst[11])
        self.entropy = float(lst[12])
        self.slope = float(lst[13])
        self.concavity = float(lst[14])
        self.answer = float(lst[15])
        self.symbol = symbol

    def get_string(self):
        if (self.var != None) and (self.entropy != None) and (self.slope != None) and \
           (self.concavity != None) and (self.dVolume != None) and (self.answer != None):
            s = str(self.index) + '\t' + \
                str(self.date) + '\t' + \
                str(self.closep)+ '\t' + \
                str(self.openp) + '\t' + \
                str(self.high) + '\t' + \
                str(self.low) + '\t' + \
                str(self.volume) + '\t' + \
                str(self.dVolume) + '\t' + \
                str(self.percentchange) + '\t' + \
                str(self.dollarchange) + '\t' + \
                str(self.spread) + '\t' + \
                str(self.var) + '\t' + \
                str(self.entropy) + '\t' + \
                str(self.slope) + '\t' + \
                str(self.concavity) + '\t' + \
                str(self.answer) + \
                '\n'
            return s
        else:
            print("Additional info: ", self.var, self.entropy, self.slope, \
                  self.concavity, self.dVolume, self.answer)
            raise Exception("Incomplete rowObject in " + self.symbol + \
                            " - something is missing in this row")


def get_rows(stockname):
    rows = []
    with open(db_dir + stockname + '-TS-full.dat', 'r') as f:
        f.readline()
        for line in f:
            rows.append(rowObjectFull(line, stockname))
    return rows


def row2input_layer(row):
    lst = [row.dVolume, 
           row.percentchange, 
           row.spread,
           row.var,
           row.entropy,
           row.slope,
           row.concavity,
           row.answer]
    return lst


def get_all_rows(names):
    rows = []
    for name in names:
        rows.extend(get_rows(name))
    return rows    


def get_sample_rows(k_samples, rows):
    if k_samples == 1:
        return np.random.choice(rows, k_samples)[0]
    else:
        return list(np.random.choice(rows, k_samples))


def normalize(lst):
    if isinstance(lst, list):
        mean = sum(lst)/len(lst)
        result = [(i-mean)/np.var(lst) for i in lst]
        return result
    else:
        raise Exception('method normalize() expected list as argument, got ' + str(type(lst)))


def readWeights(filename):
    with open(filename, 'r') as f:
        
        #read the layer sizes
        l1 = f.readline().split()
        n_input = int(l1[0])
        n_p = int(l1[1])
        n_q = int(l1[2])
        n_out = int(l1[3])
        f.readline()
        lines = f.readlines()
        
        #set weight matrix sizes
        w1 = [[0 for k in range(n_p)] for m in range(n_input)]
        w2 = [[0 for j in range(n_q)] for k in range(n_p)]
        w3 = [[0 for i in range(n_out)] for j in range(n_q)]
        
        #read w1
        for i, row in enumerate(lines[0:n_input]):
            for j, column in enumerate(row.split()):
                w1[i][j] = float(column)

        #read w2
        for i, row in enumerate(lines[n_input+1:n_input+n_p+1]):
            for j, column in enumerate(row.split()):
                w2[i][j] = float(column)

        #read w3
        for i, row in enumerate(lines[n_input+n_p+2:n_input+n_p+n_q+2]):
            for j, column in enumerate(row.split()):
                w3[i][j] = float(column)

        return n_input, n_p, n_q, n_out, w1, w2, w3


def writeWeights(w1, w2, w3, filename):
    n_input = len(w1)
    n_p = len(w2)
    n_q = len(w3)
    n_out = len(w3[0])
    header = str(n_input) + '\t' + str(n_p) + '\t' + str(n_q) + '\t' + str(n_out) + '\n'

    with open(filename, 'w') as f:
        f.write(header)
        f.write('\n')
        [f.write("\t".join([str(j) for j in i] + ['\n'])) for i in w1]
        f.write('\n')
        [f.write("\t".join([str(j) for j in i] + ['\n'])) for i in w2]
        f.write('\n')
        [f.write("\t".join([str(j) for j in i] + ['\n'])) for i in w3]


