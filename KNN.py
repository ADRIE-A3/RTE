import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.special import gamma, digamma
import pandas as pd
from numpy import genfromtxt
from datetime import datetime
from datetime import timedelta
from datetime import date
import calendar
import re
plt.rcParams.update({'font.size': 20})




def RE_normal(q, dim, det_sigm=1):
    return 0.5 *  np.log2( (2 * np.pi)**(dim) *det_sigm ) - (dim * np.log2(q))/ (2*(1-q))

def RE_cauchy(q):
    assert (q>0.5).all(), 'q needs to be bigger then 0.5'
    return np.log2( gamma(q-0.5) / (gamma(q) * (np.pi ** (q-0.5)))) / (1-q)

def RE_studentt(q, dim, eta, det_sigma = 1):
    assert (q * 0.5 * (eta+dim) - 0.5* dim > 0).all(), 'eta/dimension-condition for validity analytical result RE not met'
    return (1 / (1-q)) * ( np.log2(gamma( 0.5 * (eta + dim))**(q) * gamma(q * 0.5 * (eta + dim) - dim * 0.5) * (det_sigma) ** (0.5*(1-q)) * (eta*np.pi)**(0.5 * dim * (1-q))) - np.log2(gamma( 0.5 * eta)**(q) * gamma(q * 0.5 * (eta + dim))))


def coordinates(x, m):
    D = [x[i:i + m] for i in range(len(x) - m + 1)]
    return distance.cdist(D, D, 'euclidean')


def k_nD(i, k, M):
    D = M[i].tolist()
    D.sort()
    return D[k]


def Renyi_Estimator(k, m, q, D):
    V = np.pi ** (m / 2) / (gamma(m / 2 + 1))
    Distance_matrix = distance.cdist(D, D, 'euclidean')  # Distance_matrix = coordinates(x,m)
    (N, N) = np.shape(Distance_matrix)

    I = []
    if q !=1:
        C = (gamma(k) / (gamma(k + 1 - q))) ** (1 / (1 - q))
        for i in range(N):
            G = (N - 1) * C * V * (k_nD(i, k, Distance_matrix)) ** (m)
            I.append(G ** (1 - q))
        H = np.log2(np.sum(I) / N) / (1 - q)
        T = m / 2 * np.log2(2 * np.pi) - m / (2 * (1 - q)) * np.log2(q)
        return H
    else:
        for i in range(N):
            E = (N-1)* np.e ** (-digamma(k)) * V * (k_nD(i, k, Distance_matrix)) ** (m)
            I.append(np.log2(E))
        H = np.sum(I)/N
        return H



def RE_mean_std( m,q, D,nmin = 40, nmax = 50):
    res = np.array([Renyi_Estimator(l,m,q, D) for l in range(nmin, nmax + 1)])
    re_mean = np.sum(res)/(nmax-nmin+1)
    re_std = np.sqrt(np.sum((res-re_mean)**2)/(nmax-nmin))
    return (re_mean, re_std)



def plot_test_KKN_estimator(filename, dim, N):

    #normal distr check
    q_arr_norm = np.arange(0.1,3.1,0.1)
    norm = np.random.normal(0, 1, size = (N,dim))
    H_arr_norm = np.array([Renyi_Estimator(50,dim, qs, norm) for qs in q_arr_norm])

    xn = np.linspace(0,3.1, 1000)
    yn = RE_normal(xn, dim)

    #gauchy distr check
        #only for dimension = 1 , other dimensions: see student t-distr
    q_arr_cauch = np.arange(0.51, 3.11, 0.1) # for gauchy q_start > 0.5 , analytical result only valid in this case
    gauch = np.random.standard_cauchy((N, 1))
    H_arr_cauch = np.array([Renyi_Estimator(50, 1, qs, gauch) for qs in q_arr_cauch])

    xc = np.linspace(0.51, 3.11, 1000)
    yc = RE_cauchy(xc)

    #student-t distr check
    eta = 4 #degrees of freedom
    q_start = (dim / (dim + eta)) + 0.01 # for student-t q_start > dim / (eta + dim) , analytical result only valid in this case
    q_arr_studt = np.arange( q_start, 3.11, 0.1)
    studt = np.random.standard_t( eta, size=(N, dim))
    H_arr_studt = np.array([Renyi_Estimator(50, dim, qs, studt) for qs in q_arr_studt])

    xst = np.linspace(q_start, 3.11, 1000)
    yst = RE_studentt(xst, dim, eta)

    #uniform distr check
    q_arr_uniform = np.arange(0.1 , 3.1, 0.1)
    uniform = np.random.uniform(0, N, size=(N, dim))
    H_arr_uniform = np.array([Renyi_Estimator(50, dim, qs, uniform) for qs in q_arr_uniform])



    plt.figure(figsize=((12, 7)))

    plt.scatter(q_arr_norm, H_arr_norm, marker = 'x' ,  color= 'red')
    plt.plot(xn, yn , label  = 'Normal distribution', color  = 'red')

    plt.scatter(q_arr_cauch, H_arr_cauch, marker='x', color='blue')
    plt.plot(xc, yc, label='Cauchy distribution (1D)', color='blue')

    plt.scatter(q_arr_studt, H_arr_studt, marker='x',  color='green')
    plt.plot(xst, yst, label='Student-t distribution', color='green')

    plt.scatter(q_arr_uniform, H_arr_uniform, marker='x', color='black')
    plt.plot( np.linspace(0, 3.11, 1000), [dim * np.log2(N) for i in range(1000)], label='Uniform distribution', color='black')

    plt.xlabel('q')
    plt.ylabel('RE (bits)')
    plt.title(f'Leonenko KNN Estimator test for {dim}D distributions')
    plt.legend()
    plt.savefig(filename)



def RTE(xn, yn,q , m=1 ,l=1 , k=50):
    xn1_xm_yl, xm_yl, xn1_xm, xm = make_vectors(xn, yn, m,l)
    term1 = Renyi_Estimator(k, len(xn1_xm[0]), q, xn1_xm) - Renyi_Estimator(k, len(xm[0]), q, xm)
    term2 = Renyi_Estimator(k, len(xn1_xm_yl[0]),q, xn1_xm_yl ) + Renyi_Estimator(k, len(xm_yl[0]),q, xm_yl )
    return term1-term2

def ERTE(xn, yn,q , m=1 ,l=1 , k=50):
    rte = RTE(xn, yn, q, m, l, k)
    np.random.shuffle(yn)
    rte_shuff = RTE(xn, yn, q, m, l, k)
    return rte-rte_shuff



#makes the vectors needed for the renyi transfer entropy, given two time series and the memory m and l
def make_vectors(xn, yn, m, l):
    N = len(xn)
    assert N == len(yn), 'both timeseries should have same length'
    ml = max(m,l)
    xnms = []
    for i in range(m + 1):
        xnms.append(xn[ml - i:N - i])

    xn1_xm = list(zip(*xnms))
    xm = list(zip(*xnms[1:]))

    for i in range(l):
        xnms.append(yn[ml - 1 - i:N - 1 - i])

    xn1_xm_yl = list(zip(*xnms))
    xm_yl = list(zip(*xnms[1:]))

    return (xn1_xm_yl, xm_yl, xn1_xm, xm)


#filter data from al datetime points to only tradingdays and tradinghours of september 2021  , i.e. only weekdays between 15:30 and 22:04
#round data to minute resolution
#return dictionary with keys = daytime objects and values the filterd stock price
def cleandata(df):
    opening = '2021-9-1 15:30:00.000000'
    open_dt = datetime.strptime(opening, '%Y-%m-%d %H:%M:%S.%f')
    closing = '2021-9-30 22:05:00.000000'
    closing_dt = datetime.strptime(closing, '%Y-%m-%d %H:%M:%S.%f')
    tot_ts = df.iloc[:, 3:].values.tolist()
    ts_dict = {datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S.%f').replace(second=0,microsecond=0):[x[1]] for x in tot_ts if datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S.%f').weekday() < 5 and open_dt.date() <= datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S.%f').date() <= closing_dt.date() and open_dt.time() <= datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S.%f').time() <= closing_dt.time() }
    return ts_dict

#merge the data of time series dictionaries
#if a key of dict1 is not in dict2 the value of the stock price one minute earlier will be added
def merge_dicts(dict1, dict2):
    for k in dict1.keys():
        if k in dict2:
            dict1[k].append(dict2[k][0])
        else:
            dict1[k].append(dict2[k - timedelta(minutes=1)][0])

def ts_to_logreturn(ts):
    return np.log(ts[1:]/ts[:-1])


"""
df_apple = pd.read_csv(f'apple8to10_21.csv', sep = ',')
df_apple= df_apple.iloc[::-1]
df_sp = pd.read_csv(f'S&P8to10_21.csv', sep = ',')
df_sp= df_sp.iloc[::-1]
apple_dict = cleandata(df_apple)
sp_dict = cleandata(df_sp)


merge_dicts(sp_dict, apple_dict)
sp_arr = np.array(list(sp_dict.values()))[:,0]
apple_arr = np.array(list(sp_dict.values()))[:,1]
sp_arr = np.array(list(sp_dict.values()))[:,0]


apple_arr = ts_to_logreturn(apple_arr)
sp_arr = ts_to_logreturn(sp_arr)


print(RTE(sp_arr, apple_arr, 1.4 , m=4 ,l=1 , k=50))
print(RTE(apple_arr,sp_arr,  1.4, m=4 ,l=1 , k=50))
"""