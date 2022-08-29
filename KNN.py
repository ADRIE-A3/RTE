
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
import time
plt.rcParams.update({'font.size': 15})

import math
import numpy as np
from timebudget import timebudget
from multiprocessing import Pool

import seaborn as sns; sns.set_theme()
sns.set(font_scale = 2)

# analytical value for normal distribution at given q, dimension and correlation matrix determinant (default = 1)
def RE_normal(q, dim, det_sigm=1):
    return 0.5 *  np.log2( (2 * np.pi)**(dim) *det_sigm ) - (dim * np.log2(q))/ (2*(1-q))


# analytical value for normal distribution at given q
def RE_cauchy(q):
    assert (q>0.5).all(), 'q needs to be bigger then 0.5'
    return np.log2( gamma(q-0.5) / (gamma(q) * (np.pi ** (q-0.5)))) / (1-q)


# analytical value for normal distribution at given q, dimension, eta (dof of the distribution), correlation matrix determinant (default = 1)
def RE_studentt(q, dim, eta, det_sigma = 1):
    assert (q * 0.5 * (eta+dim) - 0.5* dim > 0).all(), 'eta/dimension-condition for validity analytical result RE not met'
    return (1 / (1-q)) * ( np.log2(gamma( 0.5 * (eta + dim))**(q) * gamma(q * 0.5 * (eta + dim) - dim * 0.5) * (det_sigma) ** (0.5*(1-q)) * (eta*np.pi)**(0.5 * dim * (1-q))) - np.log2(gamma( 0.5 * eta)**(q) * gamma(q * 0.5 * (eta + dim))))

#not used
def coordinates(x, m):
    D = [x[i:i + m] for i in range(len(x) - m + 1)]
    return distance.cdist(D, D, 'euclidean')

#given a vector in phasespace and the distance matrix between all the points in phase space,
# this function returns the k-th neirest neighbour of the given vector
def k_nD(i, k, M):
    D = M[i].tolist()
    D.sort()
    return D[k]

#this estimator returns the RE for given k, m (dimension), q and D (vectorlist), using the Leonenko estimator

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
        #T = m / 2 * np.log2(2 * np.pi) - m / (2 * (1 - q)) * np.log2(q)
        return H
    else:
        for i in range(N):
            E = (N-1)* np.e ** (-digamma(k)) * V * (k_nD(i, k, Distance_matrix)) ** (m)
            I.append(np.log2(E))
        H = np.sum(I)/N
        return H


#this function calculates the mean value and std of the RE estimated with the Leonenko estimator for different values of k
def RE_mean_std( m,q, D,nmin = 40, nmax = 50):
    res = np.array([Renyi_Estimator(l,m,q, D) for l in range(nmin, nmax + 1)])
    re_mean = np.sum(res)/(nmax-nmin+1)
    re_std = np.sqrt(np.sum((res-re_mean)**2)/(nmax-nmin))
    return (re_mean, re_std)


#plots the results of the Leonenko estimator and compares it to the analytical solutions
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


#this function calculates the RTE as a sum of 4 terms of RE of joint distributions
def RTE(xn, yn,q , m=1 ,l=1 , k=50):
    xn1_xm_yl, xm_yl, xn1_xm, xm = make_selfconditional_vectors(xn, yn, m,l)
    term1 = Renyi_Estimator(k, len(xn1_xm[0]), q, xn1_xm) - Renyi_Estimator(k, len(xm[0]), q, xm)
    term2 = Renyi_Estimator(k, len(xn1_xm_yl[0]),q, xn1_xm_yl ) - Renyi_Estimator(k, len(xm_yl[0]),q, xm_yl )
    #print(term1, term2)
    return term1-term2


def RTE_ext(xn, yn ,an, bn , cn, dn, q, m=1, l=1, k= 50):
    xn1_xm_yl, xm_yl, xn1_xm, xm = make_externalconditional_vectors(xn, an, bn, cn, dn, yn, m, l )
    term1 = Renyi_Estimator(k, len(xn1_xm[0]), q, xn1_xm) - Renyi_Estimator(k, len(xm[0]), q, xm)
    term2 = Renyi_Estimator(k, len(xn1_xm_yl[0]),q, xn1_xm_yl ) - Renyi_Estimator(k, len(xm_yl[0]),q, xm_yl )
    #print(term1, term2)
    return term1-term2



# this fucntion calculates the ERTE by subtracting the RTE with the RTE_shuffeled ,
# where the in the RTE_shuffeled the source series is shuffeld
#this takes into account the finite size effects
def ERTE(xn, yn,q , m=1 ,l=1 , k=50, N=1):
    rte = RTE(xn, yn, q, m, l, k)
    ERTE_list = np.zeros(N)
    for i in range(N):
        yn_shuff = np.random.permutation(yn)
        rte_shuff = RTE(xn, yn_shuff, q, m, l, k)
        ERTE_list[i] = rte-rte_shuff

    return (np.mean(ERTE_list), np.std(ERTE_list))



def test_ts(ts):
    print(f'adf test:')
    print(adfuller(ts))



def writetofile_ERTE_m_depence(file,ts_t, ts_s,q, m_values, k=50):
    with open(f'./data/apple_sp/ERTE_m_q={q}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'mean ERTE', 'std ERTE'])
        for m in m_values:
            avg, std =  ERTE(ts_t, ts_s, q, m, l=1, k=k)
            writer.writerow([m, avg, std])


def plot_ERTE_m_depence(file, figname):
    Lines = []
    with open(file, 'r') as f:
        cr = csv.reader(f)
        for line in cr:
            Lines.append(line)
    lines = np.array(Lines)
    m_arr = lines[1:, 0].astype(float)
    ERTE_a = lines[1:, 1].astype(float)
    ERTE_std = lines[1:, 2].astype(float)
    plt.figure(figsize=((12, 7)))

    plt.errorbar( m_arr, ERTE_a, ERTE_std, fmt='o')
    plt.savefig(f'./figures/{figname}.png')


#makes the vectors needed for the renyi transfer entropy, given two time series and the memory m and l
def make_selfconditional_vectors(xn, yn, m, l):
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




#returns two lists: 1) stringnames of the stocks 2) according timeseries
def get_timeseries(filename):
    data = pd.read_csv(f'./clean_timeseries/{filename}.csv')
    timeseries = []
    for ts in list(data.columns)[1:]:
        timeseries.append(np.array(data[ts].tolist()))
    return (list(data.columns[1:]), timeseries)

def get_RTEs_shuffeld(filename):
    data = pd.read_csv(f'./data/{filename}.csv')
    RTEs_shuffeld = []
    for rte_q in list(data.columns):
        RTEs_shuffeld.append(np.array(data[rte_q].tolist()))
    return RTEs_shuffeld


def write_RTE(filename,xn, yn ,m, l, q_range ,N=1, shuffel = False ):

    if shuffel:
        total_filename= f'./data/shuffeld_{filename}_m{m}_l{l}_N{N}_.csv'
    else:
        total_filename = f'./data/{filename}_m{m}_l{l}_N{N}_.csv'

    with open(total_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [f'q={q}' for q in q_range]
        writer.writerow(headers)
        for k in range(5,51):
            for i in range(N):
                row = []
                for q in q_range:
                    if shuffel:
                        yn_shuff = np.random.permutation(yn)
                        row.append(RTE(xn, yn_shuff, q, m, l, k))
                    else:
                        row.append(RTE(xn, yn, q, m, l, k))
                writer.writerow(row)


def write_RTE_conditioning_companies(filename,xn, yn ,m, l, q_range , othertimeseries ,N=1,shuffel= False):
    an, bn, cn, dn = othertimeseries[0], othertimeseries[1], othertimeseries[2], othertimeseries[3]
    if shuffel:
        total_filename = f'./data/shuffeld_{filename}_m{m}_l{l}_N{N}.csv'
    else:
        total_filename = f'./data/{filename}_m{m}_l{l}_N{N}.csv'

    with open(total_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [f'q={q}' for q in q_range]
        writer.writerow(headers)
        for k in range(5,51):
            for i in range(N):
                row = []
                for q in q_range:
                    if shuffel:
                        yn_shuff = np.random.permutation(yn)
                        row.append(RTE(xn, yn_shuff, q, m, l, k))

                    else:
                        row.append(RTE(xn, yn, q, m, l, k))

                writer.writerow(row)



#stocks_i = strings of stock names , timeseries_i = np_array of data of the timeseries,
#ln_... are the logreturn timeseries
stocks_z, timeseries_z = get_timeseries('zlata_timeseries')
stocks_n, timeseries_n = get_timeseries('narayan_timeseries')
ln_stocks_z, ln_timeseries_z = get_timeseries('logreturn_zlata_timeseries')
ln_stocks_n, ln_timeseries_n = get_timeseries('logreturn_narayan_timeseries')


ln_SP500 = ln_timeseries_z[0]
ln_SP5 = ln_timeseries_n[5]

#shift timeseries narayan
ln_timeseries_n_shifted = [ts[1:] for ts in ln_timeseries_n[:-1]]
ln_timeseries_n_shifted.append(ln_timeseries_n[-1][:-1])

#shift timeseries zlata
ln_timeseries_z_shifted = [ln_timeseries_n[0][:-1]]
for ts in ln_timeseries_n[1:]:
    ln_timeseries_z_shifted.append(ts[1:])


@timebudget
def create_RTE_files(para):
    i, j , shuffeld, m, l  = para[0], para[1], para[2], para[3], para[4]
    print('source series:',ln_stocks_z[i],'target series:', ln_stocks_z[j], 'm=l=', m, l, 'shuffeld:' ,shuffeld)
    """
    othertimeseries = []
    for p in range(5):
        if p !=i:
            othertimeseries.append(ln_timeseries_n_shifted[p])
            #print(ln_stocks_n[p])
    """
    write_RTE(f'RTE_{ln_stocks_z[i]}_{ln_stocks_z[j]}', ln_timeseries_z_shifted[j], ln_timeseries_z_shifted[i], m=m, l=l, q_range=[0.8, 1, 1.4], N=1, shuffel= shuffeld)

#create_RTE_files(True, 1)

@timebudget
def run_multiple_RTEfiles(operation, input, pool):
    pool.map(operation, input)



processes_count = 8

"""if __name__ == '__main__':
    processes_pool = Pool(processes_count)
    run_multiple_RTEfiles(create_RTE_files, [(0, 1 , True, 1,1) ,(0, 2 , True, 1,1), (1, 0 , True, 1,1),( 2 ,0 , True , 1,1), (0, 1 , False, 1,1) ,(0, 2 , False, 1,1), (1, 0 , False, 1,1),( 2 ,0 , False , 1,1) ] ,processes_pool)

"""

def heatmap(filename, external = False, meanorder = True) :
    coeff = 4.949733721649842* np.array([ 7.109109,3.121333, 2.005476, 5.926995, 2.040194])/100
    df_stock_sp5 = pd.DataFrame()
    df_stock_sp5_err = pd.DataFrame()
    product_dict = {}

    parameters = []
    for i,stock in enumerate(ln_stocks_n[:-1]):
        print(i, stock)
        ERTEs = []
        ERTEs_err = []

        if meanorder:
            product_dict[stock] = np.mean(timeseries_n[i]) * coeff[i]
        else:
            product_dict[stock] = coeff[i]*np.std(timeseries_n[i])

        for ml in [(1,1), (5,1), (5,5)]:
            m, l= ml[0], ml[1]

            if external:
                data = pd.read_csv(f'./data/ERTEs_stock_SP5_externalconditional_m{m}l{l}/RTE_{stock}_SP5 adjusted_m{m}_l{l}_N1_externalconditional.csv')
                data_shuff = pd.read_csv(f'./data/ERTEs_stock_SP5_externalconditional_m{m}l{l}/shuffeld_RTE_{stock}_SP5 adjusted_m{m}_l{l}_N1_externalconditional.csv')
            else:
                data = pd.read_csv(f'./data/ERTEs_SP5_stock_m{m}l{l}/RTE_SP5 adjusted_{stock}_m{m}_l{l}_N1_.csv')
                data_shuff = pd.read_csv(f'./data/ERTEs_SP5_stock_m{m}l{l}/shuffeld_RTE_SP5 adjusted_{stock}_m{m}_l{l}_N1_.csv')


           
  

            for rte_q in list(data.columns):
                rtes = np.array(data[rte_q].tolist())
                rtes_shuff = np.array(data_shuff[rte_q].tolist())
                mean = np.mean(rtes) - np.mean(rtes_shuff)
                std = np.std(rtes) + np.std(rtes_shuff)
                ERTEs.append(mean)
                ERTEs_err.append(std)


                if i==0:
                    parameters.append(f'S&P5->Comp ({m},{l}) {rte_q}')
                    #parameters.append(f'({m},{l}) {rte_q}')
            """
            
            data = pd.read_csv(f'./data/ERTEs_stock_SP5_m{m}l{l}/RTE_{stock}_SP5 adjusted_m{m}_l{l}_N1_.csv')
            data_shuff = pd.read_csv(f'./data/ERTEs_stock_SP5_m{m}l{l}/shuffeld_RTE_{stock}_SP5 adjusted_m{m}_l{l}_N1_.csv')

            for rte_q in list(data.columns):
                rtes = np.array(data[rte_q].tolist())
                rtes_shuff = np.array(data_shuff[rte_q].tolist())
                mean = np.mean(rtes) - np.mean(rtes_shuff)
                std = np.std(rtes) + np.std(rtes_shuff)
                ERTEs.append(mean)
                ERTEs_err.append(std)
                if i == 0:
                    parameters.append(f'Comp->S&P5 ({m},{l}) {rte_q}')
                    # parameters.append(f'm={m} l={l} {rte_q}')
            """
        df_stock_sp5[stock] = ERTEs
        df_stock_sp5_err[stock] = ERTEs_err


    df_stock_sp5['parameters (m,l)'] = parameters
    df_stock_sp5.set_index("parameters (m,l)", inplace = True)
    new_row = pd.Series(data=product_dict, name='coeff.meanprize')
    df_stock_sp5 = df_stock_sp5.append(new_row, ignore_index=False)
    df_stock_sp5 = df_stock_sp5.sort_values(by ='coeff.meanprize', axis=1)

    print(df_stock_sp5)


    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(20, 13))
    sns.heatmap(df_stock_sp5.iloc[:-1], annot=True, linewidths=.5, ax=ax)
    if external:
        plt.title(f'ERTE with external conditioning')

    else:
        plt.title(f'ERTE (m,l)=({m},{l}) with self conditioning')
    plt.tight_layout()
    plt.savefig(filename)

heatmap('./figures/ERTE_ext_cond_stdorder', external= True, meanorder=False )

"""for stock in ln_stocks_n[:-1]:
    data = pd.read_csv(f'./data/RTE_SP5 adjusted_{stock}_m5_l1_N1_.csv')
    data_shuff = pd.read_csv(f'./data/shuffeld_RTE_SP5 adjusted_{stock}_m5_l1_N1_.csv') 

    print('T -> SP5 --> ', stock, ', (m,l) = (5,1) : selfconditional')
    print('------------------------------------------------------------------------------------------------------')
    for rte_q in list(data.columns):
        print(rte_q)
        rtes = np.array(data[rte_q].tolist())
        rtes_shuff = np.array(data_shuff[rte_q].tolist())
        mean = np.mean(rtes) - np.mean(rtes_shuff)
        std = np.std(rtes) + np.std(rtes_shuff)
        print(mean, std)
    print('------------------------------------------------------------------------------------------------------')
"""

"""

for stock in ln_stocks_n[:-1]:
    data = pd.read_csv(f'./data/RTE_SP5 adjusted_{stock}_m5_l1_N1_.csv')

    data_shuff = pd.read_csv(f'./data/shuffeld_RTE_SP5 adjusted_{stock}_m5_l1_N1_.csv')

    print('T -> SP5 --> ', stock, ', (m,l) = (5,1) : selfconditional')
    print('------------------------------------------------------------------------------------------------------')
    for rte_q in list(data.columns):
        print(rte_q)
        rtes = np.array(data[rte_q].tolist())
        rtes_shuff = np.array(data_shuff[rte_q].tolist())
        mean = np.mean(rtes) - np.mean(rtes_shuff)
        std = np.std(rtes) + np.std(rtes_shuff)
        print(mean, std)
    print('------------------------------------------------------------------------------------------------------')

"""

"""stock = ln_stocks_n[0]

data = pd.read_csv(f'./data/RTE_{stock}_SP5_historySP5_juist_m1_l1_N1_.csv')
data_shuff = pd.read_csv(f'./data/RTE_shuff_{stock}_SP5_historySP5_m1_l1_N1_.csv')
data_shuff_juist = pd.read_csv(f'./data/RTE_shuff_{stock}_SP5_historySP5_juist_m1_l1_N1_.csv')
print('T', stock, '-> S&P5, (m,l) = (1,1)')
print('------------------------------------------------------------------------------------------------------')
for rte_q in list(data.columns):
    print(rte_q)
    rtes = np.array(data[rte_q].tolist())
    rtes_shuff = np.array(data_shuff[rte_q].tolist())
    mean = np.mean(rtes) - np.mean(rtes_shuff)
    std = np.std(rtes) + np.std(rtes_shuff)
    print('fout:', mean, std)
    rtes_shuff = np.array(data_shuff_juist[rte_q].tolist())
    mean = np.mean(rtes) - np.mean(rtes_shuff)
    std = np.std(rtes) + np.std(rtes_shuff)
    print('juist', mean, std)
print('------------------------------------------------------------------------------------------------------')
"""


"""SP500 = timeseries_z[0]
SP5= timeseries_n[5]
print(len(SP5))
print(len(SP500))

plt.figure(figsize=((12, 7)))
plt.plot(range(len(SP500)), 100*SP500/SP500[0],  color = 'red', label = 'Emprical S&P500')
plt.plot(range(len(SP5)), 100*SP5/SP5[0] ,color  = 'blue' ,label = 'Constructed S&P5')
plt.plot(range(len(timeseries_n[1])), 100*timeseries_n[1]/timeseries_n[1][0], color = 'green', label = 'Apple')
plt.legend()
plt.title('S&P500, S&P5 and Apple time series ')
plt.xlabel('Time (minutes)')
plt.ylabel('Normalised Index Value')

plt.savefig(f'./figures/SP5_SP500_apple_comparison.png')

"""

"""print(len(ln_timeseries_n[0]), len(ln_timeseries_z[1]))
plt.figure(figsize=((12, 7)))
plt.plot(range(len(ln_timeseries_n[0])), ln_timeseries_n[0],  color = 'red', label = stocks_n[0])
plt.plot(range(len(ln_timeseries_z[1])), ln_timeseries_z[1],color  = 'blue' ,label = stocks_z[1])
plt.legend()
plt.title('comparisson logreturn apple data narayan and zlata')
plt.savefig(f'./figures/logreturn_apple_comparison.png')

"""

"""
m_range = [1,5,10,15,20,25,30,35,40,45,50, 55,60,65]
for q in [0.8, 1,1.4,1.8]:
    #writetofile_ERTE_m_depence(f'./data/apple_sp/ERTE_m_q={q}.csv', sp, ap, q,m_range  )
    plot_ERTE_m_depence(f'./data/apple_sp/ERTE_m_q={q}.csv', f'test_m_dependence_q={q}')"""

