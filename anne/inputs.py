#!/usr/bin/python3

#import packages
import numpy as np
from scipy.stats import entropy as entrop
from numpy import var as npvar
from numpy import histogram as nphistogram



class inputs():

    """
    This class is for the computing of values to be sent to the input
    layer of my ANN. Other values to be sent to the input layer can be
    pulled straight from the yahoo-finance data, i.e., close price or volume
    """

    def __init__(self):
        self._nvar = 10 #number of days to use for computing the variance
        self._nent = 30 #number of days to use for computing the entropy
        self._disc_ent = 5 #discretization bins for entropy calculation
        self._d1_order = 3 #order of approximation for 1st order backward FD
        self._d2_order = 3 #order of approximation for 2nd order backward FD
        self._h = 1 #width of steps in time series - for stock prices it is one day


    def onebfd(self, price_list): 

        """
        This calculates the 1st order backwards finite difference to 
        1st, 2nd, or 3rd order accuracy - tested and correct
        """

        self._check_list_input('onebfd', self._d1_order+1, price_list)        
        p = price_list
        if self._d1_order == 1:
            result = (p[-1] - p[-2]) / self._h
            return result
        elif self._d1_order == 2:
            result = (3*p[-1] - 4*p[-2] + p[-3]) / (2*self._h)
            return result
        elif self._d1_order == 3:
            result = (11*p[-1] - 18*p[-2] + 9*p[-3] - 2*p[-4]) / (6*self._h)
            return result


    def twobfd(self, price_list):

        """
        This calculates the 2nd order backwards finite difference to 
        1st, 2nd, or 3rd order accuracy - tested and correct
        """

        self._check_list_input('twobfd', self._d2_order+2, price_list)
        p = price_list
        if self._d2_order == 1:
            result = (p[-1] - 2*p[-2] + p[-3]) / self._h**2
            return result
        elif self._d2_order == 2:
            result = (2*p[-1] - 5*p[-2] + 4*p[-3] - p[-4]) / self._h**2
            return result
        elif self._d2_order == 3:
            result = (35*p[-1] - 104*p[-2] + 114*p[-3] - 56*p[-4] + 11*p[-5]) / (12*self._h**2)
            return result


    def percent_change(self, open_price, close_price):
        result = (close_price - open_price) / open_price
        return result


    def var(self, price_list):

        """
        This calculates the variance of a time series over the last
        self._nvar days
        """

        self._check_list_input('var', self._nvar, price_list)
        tmp_list = price_list[-self._nvar:]
        result = npvar(tmp_list)
        return result


    def entropy(self, price_list):

        """
        This calculates the entropy of a time series over the last
        self._nent days where the range of values is binned into
        self._disc_ent bins ranging from min(p) to max(p)
        """

        self._check_list_input('entropy', self._nent, price_list)
        tmp_list = price_list[-self._nent:]
        freq = list(nphistogram(tmp_list, bins=self._disc_ent)[0])
        tot = sum(freq)
        freq = [i / tot for i in freq]
        result = entrop(freq)
        return result


    def _check_list_input(self, method_name, min_size, lst):
        if len(lst) < min_size:
            raise Exception("Expected list argument of length " +str(min_size) + \
                            " in input." + method_name + "(): Too short")
        elif not isinstance(lst, list):
            raise Exception("Expected list argument of length " +str(min_size) + \
                            " in input." + method_name + "(): Not a list")

