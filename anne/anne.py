#!/usr/bin/python3

#import packages
import numpy as np
from TSTools import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl

#parameters
n_input = 7
beta = 1.0
eta = .1
db_dir = '/home/wes/Stock/analysis/anne/db/'
names = ['WMT', 'FMC', 'VZ', 'HSE.TO', 'CL', 'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY', \
         'MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', \
         'DLX', 'GK', 'PACW', 'AMZN', 'PCLN', 'TRIP', 'ALB', 'TWOU', 'AYA', 'TRU']
names = ['MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', 'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', \
         'DLX', 'GK', 'PACW', 'AMZN', 'PCLN', 'TRIP', 'ALB', 'TWOU', 'TRU']




def AnneForward(w1, w2, activation_function, input_layer, target):

    """
    w1 and w2 are the weights of the input/hidden and hidden/output edges, respectively
    activation_function is, for example, a sigmoid for mapping input to [0,1)
    input_layer is a list of input values

    returns result on [0,1) and correct answer {0,1} - 0 is sell, 1 is buy
    """
    
    hidden_in = [0 for i in range(len(w2))]
    for indexi, valuei in enumerate(hidden_in):
        hidden_in[indexi] = sum([valuej*w1[indexj][indexi] for indexj,valuej in enumerate(input_layer)])
    hidden_out = [activation_function(beta, i) for i in hidden_in]
    res_in = sum([valuei*w2[indexi] for indexi, valuei in enumerate(hidden_out)])
    res_out = activation_function(beta, res_in)
    error = .5*(target-res_out)**2
    
    return input_layer, w1, hidden_in, hidden_out, w2, res_in, res_out, error, target



def AnneBackward(input_layer, w1, hidden_in, hidden_out, w2, res_in, res_out, 
                 error, target, derivative_function):

    """
    w1 and w2 are the weights of the input/hidden and hidden/output edges, respectively
    hidden_results are the results at the hidden nodes
    derivative_function is the derivative of whatever function was used as activation_function
    difference is the correction that needs to be applied to result to get right answer

    returns new weights, w1 and w2
    """

    for indexj, valuej in enumerate(w1):
        for indexi, valuei in enumerate(w2):
            w1[indexj][indexi] += eta*input_layer[indexj]*valuei*\
                                  derivative_function(beta, res_in)*\
                                  derivative_function(beta, hidden_in[indexi])*\
                                  (target-res_out)
    for indexi, valuei in enumerate(w2):
        w2[indexi] += eta*hidden_out[indexi]*derivative_function(beta, res_in)*(target-res_out)

    return w1, w2





def init_weights(n_hidden):
    #init the weights of the edges between input layer and hidden layer
    w1 = [list(np.random.uniform(0,1,n_hidden))for i in range(n_input)]
    #init the weights of the edges between hidden layer and output layer
    w2 = list(np.random.uniform(0,1,n_hidden))
    return w1, w2


def set_eta_beta(new_eta, new_beta):
    return new_eta, new_beta


def runAnneOnSet(input_layer_list):
    w1, w2 = init_weights()
    errors = []
    for row in input_layer_list:
        af = AnneForward(w1, w2, sigmoid, row[:-1], row[-1])
        errors.append(af[7])
    return errors


def set_good_weight():
    w1 = [[17.629907414882556, 11.643904500577365, -17.145313319809336, -5.1770079402309017],
          [-0.74336743763947888, 10.904662345555415, -6.4372743944895268, 2.5878584176559674],
          [-0.039680874118601724, -0.48659978317404118, 1.0663338637218169, -0.65022783739315515],
          [23.142683270835185, -7.8285271270255006, -1.6154105675672483, -15.38675281427582],
          [-14.490561517198607, -43.021974404801021, 18.2181993557144, -33.80928828218056],
          [-21.142055545023972, -16.281198236850429, 16.64313891967338, 12.19997021334083],
          [-41.740427605916636, 4.8814571291558595, 44.651738480655695, -3.2627661071788707]]
    w2 = [-1.7166738841492544, -0.38335505346348869, -1.8628959828451539, -4.2638359132653312]
    '''
    w1 = [[17.615762048911321, 11.582569328508736, -17.119524882339917, -5.1773964969377104],
          [-0.74303974889841817, 10.907467270164993, -6.4431686988277983, 2.5877025751222198],
          [-0.040550022540601617, -0.48736224734230849, 1.0673010847281439, -0.65019737542771472],
          [23.130310322305153, -7.8783246187231502, -1.6707602952602214, -15.386419324392293],
          [-14.509376924588935, -43.04559716741587, 18.238661090340955, -33.808582366907821],
          [-21.142742278004256, -16.276714540879144, 16.651282468922027, 12.199873813037593],
          [-41.745437842019683, 4.7441335851643878, 44.647714902465694, -3.2699847801913933]]
    w2 = [0.074995464114706006, 0.074101821038335222, -0.044977694023571699, -4.2640215296522035]
    '''
    return w1, w2


#===============================================#
#                    ____                       #
#                    MAIN                       #
#                                               #
#===============================================#



'''
er = []
for name in names[25:26]:
    rows = get_rows(name)
    rows = [row2input_layer(i) for i in rows]
    for row in rows[1000:1051]:
        for i in range(1000):
            af = AnneForward(w1, w2, sigmoid, row[:-1], row[-1])
            w1, w2 = AnneBackward(af[0], af[1], af[2], af[3], af[4], af[5], af[6], af[7], af[8], d_sigmoid)
            er.append(af[7])

smooth = savgol_filter(er, 51, 2)#, mode='nearest')
print(sum(er)/len(er))
plt.plot([i for i in range(len(er))], smooth)
plt.show()
'''


nit = 7000
nn = list(range(nit))
er = []
l = get_all_rows(names)
#l = [i for i in l if i.answer == 0.0]
#w1, w2 = set_good_weight()#init_weights(4)
w1,w2=init_weights(7)
eta = .01
sample = get_sample_rows(nit,l)
sample = [row2input_layer(i) for i in sample]

jit = 20
beta = 1.02
corr=0
wrong=0
for n in nn:
    row = sample[n]
    '''
    af = AnneForward(w1, w2, sigmoid, row[:-1], row[-1])
    if af[6] > .5:
        ans = 1
    else:
        ans = 0
    if ans == af[8]:
        corr+=1
    else:
        wrong+=1
    '''
    
    #if n > int(.2*nit):
    #    jit = int(80*(1-(n/nit)))+1
    for j in range(jit):
        af = AnneForward(w1, w2, sigmoid, row[:-1], row[-1])
        w1, w2 = AnneBackward(af[0], af[1], af[2], af[3], af[4], af[5], af[6], af[7], af[8], d_sigmoid)
    er.append(af[7])
    #if n > int(.5*nit):
    #    eta = .05+(1-(n/nit))
    
    print(n,af[6],af[8])

mpl.rcParams['agg.path.chunksize'] = 10000
smooth = savgol_filter(er, 301, 2, mode='nearest')
with open('4training10k.dat', 'w') as f:
    [f.write(str(i)+'\t'+str(j)+'\t'+str(k)) for i,j,k in zip(nn,er,smooth)]
plt.plot(nn,er, c='blue')
plt.plot(nn,smooth, c='red')
plt.xlabel('iteration')
plt.ylabel('error')
plt.savefig('4NodeTrainingBuy' + str(int(nit/1000)) + 'k.png', dpi=500)
[print(i) for i in w1]
print()
print(w2)

#print(corr,wrong,corr/(corr+wrong))

