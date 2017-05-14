#!/usr/bin/python3

#import packages
import numpy as np
from TSTools import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


#parameters
db_dir = '/home/wes/Stock/analysis/anne/db/'
names = ['WMT', 'FMC', 'VZ', 'HSE.TO', 'CL', \
         'TSLA', 'SQM', 'BCN.V', '002460.SZ', 'PCRFY', \
         'MMS', 'PEB', 'JCOM', 'CHSP', 'UBSI', \
         'MDP', 'SSNC', 'AOS', 'AAON', 'APOG', \
         'DLX', 'GK', 'PACW', 'AMZN', 'PCLN', \
         'TRIP', 'ALB', 'TWOU', 'AYA', 'TRU']
with open('names.txt', 'r') as f:
    names = [line.rstrip() for line in f]

names = ['SSNC']
n_input = 1
n_p = 1
n_q = 1
n_output = 2
num_samples = 10000
bias = 0.01
ww = [[[] for j in range(n_p)] for i in range(n_input)]


#functions
def mk_outfile_name():
    if num_samples > 999999:
        ns = str(num_samples/1000000)+'M'
    elif num_samples > 999:
        ns = str(num_samples/1000)+'k'
    else:
        ns = str(num_samples)
    model = str(n_input)+str(n_p)+str(n_q)+str(n_output)
    bias_s = str(bias)
    ext = '.pdf'
    return "-".join([model,bias_s,ns])+'.png'

def cooling_eta(iteration, eta_i, eta_f, max_it):
    mean = -(.99)*iteration/(max_it) + 1.0
    return np.random.gamma(mean)*np.exp(-2*iteration/max_it)


def mk_weights(n_input, n_p, n_q, n_output, kw = 'random'):
    uniform_weight = 0.5
    if kw == 'uniform':
        w1 = [[uniform_weight for k in range(n_p)] for m in range(n_input)]
        w2 = [[uniform_weight for j in range(n_q)] for k in range(n_p)]
        w3 = [[uniform_weight for i in range(n_output)] for j in range(n_q)]
    else:
        w1 = [list(np.random.uniform(0,.05,n_p)) for m in range(n_input)]
        w2 = [list(np.random.uniform(0,.05,n_q)) for k in range(n_p)]
        w3 = [list(np.random.uniform(0,.05,n_output)) for j in range(n_q)]
    return w1, w2, w3


def Anne2Forward(w1, w2, w3, input_layer, activation_function1, activation_function2, targets):
    """
    This is Anne part II.
    2 hidden layers, 2 ouput nodes, 7 input nodes, arbitrary activation function.
    This time I'll need 3 weight matrices and 2 targets on the output layer.
    Targets represent vectors; [1,0] = buy, [0,1] = sell.
    """

    p_in = [sum([w1[m][k]*input_layer[m] for m in range(len(w1))]) + bias for k in range(len(w2))]
    p_out = [activation_function1(elm) for elm in p_in]
    
    q_in = [sum([w2[k][j]*p_out[k] for k in range(len(w2))]) + bias for j in range(len(w3))]
    q_out = [activation_function2(elm) for elm in q_in]

    res_in = [sum([w3[j][i]*q_out[j] for j in range(len(w3))]) + bias for i in range(len(targets))]
    res_out = [activation_function2(elm) for elm in res_in]
    error = sum([0.5*(targets[elm] - res_out[elm])**2 for elm in range(len(res_out))])

    result = [p_in, p_out, q_in, q_out, res_in, res_out, error]
    original_input = [w1, w2, w3, input_layer, activation_function1, activation_function2, targets]

    return original_input, result


def Anne2Backward(w1, w2, w3, results, input_layer, derivative_function1, derivative_function2, targets):
    """
    docstring
    """
    s_prime1 = derivative_function1
    s_prime2 = derivative_function2
    p_in, p_out, q_in, q_out, res_in, res_out, error = results
    dw1 = [[1]*len(w2) for l in range(len(w1))]
    dw2 = [[1]*len(w3) for l in range(len(w2))]
    dw3 = [[1]*len(targets) for l in range(len(w3))]

    for m in range(len(w1)):
        for k in range(len(w2)):
            dw1[m][k] = sum([sum([(targets[i]-res_out[i])*\
                                  s_prime2(res_in[i])*\
                                  w3[j][i]*\
                                  s_prime2(q_in[j])*\
                                  w2[k][j]*\
                                  s_prime1(p_in[k])*\
                                  input_layer[m] \
                                  for j in range(len(w3))]) for i in range(len(targets))])

    for k in range(len(w2)):
        for j in range(len(w3)):
            dw2[k][j] = sum([(targets[i]-res_out[i])*\
                             s_prime2(res_in[i])*\
                             w3[j][i]*\
                             s_prime2(q_in[j])*\
                             p_out[k] for i in range(len(targets))])

    for j in range(len(w3)):
        for i in range(len(targets)):
            dw3[j][i] = (targets[i]-res_out[i])*s_prime2(res_in[i])*q_out[j]

    w1 = np.add(w1, dw1)
    w2 = np.add(w2, dw2)
    w3 = np.add(w3, dw3)

    return w1, w2, w3



def classify(w1, w2, w3):
    trading_days = get_all_rows(names)
    days = get_sample_rows(10000, trading_days)
    days = [row2input_layer(i) for i in days]
    win = 0
    lose = 0
    for day in days:
        targets = [day[-1], 1-day[-1]]
        original_input, results = Anne2Forward(w1, w2, w3, day[:-1], sigmoid, sigmoid, targets)
        o1, o2 = results[-2]
        if o1 > o2:
            result = 1
        else:
            result = 0
        if result == targets[0]:
            win += 1
        else:
            lose += 1
    print('PERCENTAGE CORRECT: ', 100*win/(win+lose))



def mkContour():
    plst = list(range(1,40))
    qlst = list(range(1,40))
    P = [plst]*len(qlst)
    Q = [[qlst[i]]*len(plst) for i in range(len(qlst))]
    trading_days = get_all_rows(names)
    days = get_sample_rows(num_samples, trading_days)
    days = [row2input_layer(i) for i in days]
    matrix1 = [[0]*len(plst) for i in range(len(qlst))]
    matrix2 = [[0]*len(plst) for i in range(len(qlst))]

    for ip, nnp in enumerate(plst):
        for iq, nq in enumerate(qlst):
            n_p = nnp
            n_q = nq
            w1, w2, w3 = mk_weights(n_input, n_p, n_q, n_output, 'random')
            eta = [cooling_eta(i, 4, .001, len(days)) for i in range(len(days))]
            error = []

            for n in range(len(days)):
                day = days[n]
                targets = [day[-1], 1-day[-1]]
                original_input, results = Anne2Forward(w1, w2, w3, day[1:4], sigmoid, sigmoid, targets)
                w1, w2, w3 = Anne2Backward(w1, w2, w3, results, day[:-1], d_sigmoid, d_sigmoid, targets)

                #[[ww[i][j].append(w1[i][j]) for j in range(n_p)] for i in range(n_input)]
                error.append(results[-1])
                #print(n,results[-1])
            matrix1[iq][ip] = np.var(error[-50:])#sum(error[-50:])/50
            matrix2[iq][ip] = sum(error[-50:])/50
            print(nnp,nq,sum(error[-50:])/50, np.var(error[-50:]))
    plt.contourf(P,Q,matrix1, 50)
    plt.colorbar()
    plt.title('sigmoid/sigmoid N='+str(num_samples)+' VAR10k')
    plt.xlabel('first hidden layer')
    plt.ylabel('second hidden layer')
    plt.savefig('matrixVarP1.png', dpi=500)
    plt.close()

    plt.contourf(P,Q,matrix2, 50)
    plt.colorbar()
    plt.title('sigmoid/sigmoid N='+str(num_samples)+' MEAN10k')
    plt.xlabel('first hidden layer')
    plt.ylabel('second hidden layer')
    plt.savefig('matrixMeanP1.png', dpi=500)
    plt.close()




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






trading_days = get_all_rows(names)
#days = get_sample_rows(num_samples, trading_days)
days = [row2input_layer(i) for i in trading_days]
'''
w1, w2, w3 = mk_weights(n_input, n_p, n_q, n_output, 'random')
#n_input, n_p, n_q, n_output, w1, w2, w3 = readWeights('weights1.txt')
eta = .5#[cooling_eta(i, 5, .0001, len(days)) for i in range(len(days))]
error = []

for n in range(len(days)):
    day = days[n]
    targets = [day[-1], 1-day[-1]]
    original_input, results = Anne2Forward(w1, w2, w3, day[2:3], sigmoid, sigmoid, targets)
    w1, w2, w3 = Anne2Backward(w1, w2, w3, results, day[:-1], d_sigmoid, d_sigmoid, targets)
    error.append(results[-1])
    print(n,results[-1])




smoothed = savgol_filter(error, 61, 2, mode='nearest')
plt.plot(list(range(len(days))), error, c='blue')
plt.plot(list(range(len(days))), smoothed, c='red')
plt.ylim(0,1)
plt.savefig(mk_outfile_name(), dpi=500)
last_avg = sum(error[-200:])/200
print(last_avg)
plt.close()
print(mk_outfile_name())
writeWeights(w1,w2,w3,'weights1.txt')

n_input, n_p, n_q, n_output, w1, w2, w3 = readWeights('weights1.txt')
classify(w1,w2,w3)
'''

s = [i[1] for i in days]
plt.plot(list(range(len(s))),s)
#plt.xlim(-2,2)
plt.savefig('hist.pdf')
