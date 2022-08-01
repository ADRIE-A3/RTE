import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.special import gamma, digamma
import pandas as pd
from numpy import genfromtxt
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
import csv
import sklearn
from sklearn import preprocessing
from datetime import date
import calendar
import re
plt.rcParams.update({'font.size': 8})

a = np.zeros(10)
b = np.ones(10)

xn = np.array([f'x{i}' for i in range(10)])
an = np.array([f'a{i}' for i in range(10)])
bn = np.array([f'b{i}' for i in range(10)])
cn = np.array([f'c{i}' for i in range(10)])
dn = np.array([f'd{i}' for i in range(10)])
yn = np.array([f'y{i}' for i in range(10)])

def make_externalconditional_vectors(xn, sn, zn, tn, un, yn, m, l ):
    N = len(xn)
    assert N == len(yn) == len(sn)== len(zn)== len(tn)== len(un), 'all timeseries should have same length'
    ml = max(m,l)
    xnms = []
    for i in range(m):
        xnms.append(sn[ml-1 - i:N-1 - i])
        xnms.append(zn[ml-1 - i:N-1 - i])
        xnms.append(tn[ml-1 - i:N -1 - i])
        xnms.append(un[ml -1 - i:N-1 - i])

    xm = list(zip(*xnms))

    for i in range(1):
        xnms.append(xn[ml - i:N - i])

    xn1_xm = list(zip(*xnms))

    xnynms = []
    for i in range(m):
        xnynms.append(sn[ml-1 - i:N-1 - i])
        xnynms.append(zn[ml-1 - i:N-1 - i])
        xnynms.append(tn[ml-1 - i:N -1 - i])
        xnynms.append(un[ml -1 - i:N-1 - i])
    for i in range(l):
        xnynms.append(yn[ml - 1 - i:N - 1 - i])

    xm_yl = list(zip(*xnynms))

    for i in range(1):
        xnynms.append(xn[ml - i:N - i])

    xn1_xm_yl= list(zip(*xnynms))


    return (xn1_xm_yl, xm_yl, xn1_xm, xm)

xn1_xm_yl, xm_yl, xn1_xm, xm = make_externalconditional_vectors(xn, an, bn, cn, dn, yn , m = 5, l=5)
print(xm)
print(len(xm[0]))
print(xn1_xm)
print(len(xn1_xm[0]))
print(xm_yl)
print(len(xm_yl[0]))
print(xn1_xm_yl)
print(len(xn1_xm_yl[0]))

