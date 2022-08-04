import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import date
import calendar
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.stattools import adfuller
import csv
import calendar
import re
import numpy as np


#familiarising with datetime objects
def familiarsing(filename):
    df = pd.read_csv(f'./data/{filename}.csv', sep = ',')
    df = df.iloc[::-1, 3:]
    date_time_str = a = df.loc[0, 'created_at']
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    opening = '2021-9-1 15:00:00.000000'
    opt_dt = datetime.strptime(opening, '%Y-%m-%d %H:%M:%S.%f')
    print ("The type of the date is now",  type(date_time_obj))
    print ("The date is", date_time_obj)
    print("the date - 1minute is", date_time_obj-timedelta( minutes=1))
    print(date_time_obj.hour)
    print(date_time_obj.weekday())
    print(date_time_obj.month)
    print(date_time_obj.minute)

    t2 = timedelta( minutes=5)
    t3 = timedelta( minutes=1)


    f = t2 / t3
    print(f)

    print(date_time_obj.date())
    print(opt_dt.date())
    print(opt_dt.date() < date_time_obj.date())

#filter data from al datetime points to only tradingdays and tradinghours of september 2021  , i.e. only weekdays between 15:30 and 22:04
#round data to minute resolution
#return dictionary with keys = daytime objects and values the filterd stock price
def cleandata(df, formatstring, timediff= timedelta( hours=0, minutes=0)):
    opening = '2021-9-1 15:31:00.000000'
    open_dt = datetime.strptime(opening, '%Y-%m-%d %H:%M:%S.%f')
    closing = '2021-9-30 22:01:00.000000'
    closing_dt = datetime.strptime(closing, '%Y-%m-%d %H:%M:%S.%f')
    special_closing_day = '2021-9-6'
    special_closing_day_dt = datetime.strptime(special_closing_day, '%Y-%m-%d')
    tot_ts = df.values.tolist()
    ts_dict = {(datetime.strptime(x[0],formatstring).replace(second=0,microsecond=0)+timediff):[x[1]] for x in tot_ts if (datetime.strptime(x[0], formatstring)+timediff).weekday() < 5 and open_dt.date() <= (datetime.strptime(x[0], formatstring)+timediff).date() <= closing_dt.date() and (datetime.strptime(x[0], formatstring)+timediff).date()!= special_closing_day_dt.date() and open_dt.time() <= (datetime.strptime(x[0], formatstring)+timediff).time() < closing_dt.time()  }
    k0 = sorted(ts_dict.keys())[0]
    for k in sorted(ts_dict.keys())[1:]:
        N = (k-k0)/timedelta(minutes=1)
        if k.date()==k0.date() and N != 1.0:
            #print(k0)
            for i in range(int(N-1)):
                #print(i+1)
                #print(float(ts_dict[k][0])-float(ts_dict[k0][0]))
                ts_dict[k0+ (i+1)*timedelta( minutes=1)] = [ts_dict[k0][0] + (i+1)* (ts_dict[k][0]-ts_dict[k0][0])/(N)]
        k0=k
    return ts_dict

def prices_to_lnreturns(ts_dict):
    ln_dict = {}
    k0 = sorted(ts_dict.keys())[0]
    for k in sorted(ts_dict.keys())[1:]:
        if k.date() == k0.date():
            assert(k-k0 == timedelta( minutes=1)), f'still timegaps at {k, k0}'
            ln_dict[k] = [np.log(ts_dict[k][0]/ts_dict[k0][0])]
        k0=k
    return ln_dict

def merge_dicts(dict1, dict2):
    for k in dict1.keys():
        if k in dict2:
            dict1[k].append(dict2[k][0])
        else:
            print(f'non appearing datatimeobject in dictionary at datetime{k}')
            dict1[k].append(dict2[k - timedelta(minutes=1)][0])


def test_ts(ts):
    print(f'adf test:')
    print(adfuller(ts))


#read files, make dataframes, clean the data of both files (with 'cleandata(df)' ) merge timepoints with target serie as reference,
# map to logreturnsvalues and return timeseries of target serie and scource serie


#read data from zlata and return twoe dictionaries 1) keys= datetimes,  values = prices 2) keys = datetimes, values= logreturns
def read_zlata_data(filename):
    df = pd.read_csv(f'./data/{filename}.csv', sep = ',')
    df= df.iloc[::-1, 3:]
    dict = cleandata(df,  '%Y-%m-%d %H:%M:%S.%f')
    ln_dict = prices_to_lnreturns(dict)
    return (dict, ln_dict)


#read data from Narayan and return twoe dictionaries 1) keys= datetimes,  values = prices 2) keys = datetimes, values= logreturns
def read_narayan_data(filename):
    Lines = []
    with open(f'./data/{filename}.csv', 'r') as f:
        cr = csv.reader(f)
        for line in cr:
            Lines.append(line)
    Dates = Lines[0]
    Close = Lines[1]

    df = pd.DataFrame()
    df['Date'] = Dates[1:]
    df['Close'] = list(map(float, Close[1:]))
    df = df[::-1]
    #make new readable file to check results etc (not necassary)
    df.to_csv(f'./data/{filename}_readable.csv', index = False, header = False, sep = ',')

    #6hour timedifference taken into account
    dict = cleandata(df,'%Y-%m-%d %H:%M:%S', timediff= timedelta( hours=6)  )
    ln_dict = prices_to_lnreturns(dict)
    return (dict, ln_dict)

def write_time_series_to_file(filename, headers,dicts):
    with open(f'./clean_timeseries/{filename}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for key in sorted(dicts[0].keys()):
            row = [key]
            for d in dicts:
                row.append(d[key][0])
            writer.writerow(row)

def SP5_constructor(dicts):
    SP5_dict = {}
    for key in sorted(dicts[0].keys()):
        sp5 =  (4.949733721649842 * 7.109109) * dicts[0][key][0] + (4.949733721649842 * 5.926995) * dicts[3][key][0] + (4.949733721649842 * 3.121333) * dicts[1][key][0]  + (4.949733721649842 * 2.005476) * dicts[2][key][0]  + (  4.949733721649842 * 2.040194) * dicts[4][key][0]
        SP5_dict[key] = [sp5/100]
        ln_SP5_dict = prices_to_lnreturns(SP5_dict)
    return (SP5_dict, ln_SP5_dict)





#read zlata data
sp_dict , sp_ln_dict = read_zlata_data('S&P8to10_21')
apple_dict , apple_ln_dict = read_zlata_data('apple8to10_21')
alpha_dict , alpha_ln_dict = read_zlata_data('alphabet8to10_21')
fb_dict , fb_ln_dict = read_zlata_data('facebook8to10_21')

prices_dicts = [sp_dict  ,apple_dict ,alpha_dict ,fb_dict]
lnreturns_dicts = [sp_ln_dict, apple_ln_dict, alpha_ln_dict, fb_ln_dict]
names = ['datetime', 'S&P500', 'Apple', 'Alphabet', 'Facebook']

write_time_series_to_file('zlata_timeseries', names , prices_dicts)
write_time_series_to_file('logreturn_zlata_timeseries', names , lnreturns_dicts)



#read Narayan data
AAPL2_dict,  AAPl2_ln_dict = read_narayan_data('AAPL2 raw')
AMZN2_dict,  AMZN2_ln_dict = read_narayan_data('AMZN2 raw')
GOOGL2_dict,  GOOGL2_ln_dict = read_narayan_data('GOOGL2 raw')
MSFT2_dict,  MSFT2_ln_dict = read_narayan_data('MSFT2 raw')
TSLA2_dict,  TSLA2_ln_dict = read_narayan_data('TSLA2 raw')

AAPL1_dict,  AAPl1_ln_dict = read_narayan_data('AAPL1 raw')
AMZN1_dict,  AMZN1_ln_dict = read_narayan_data('AMZN1 raw')
GOOGL1_dict,  GOOGL1_ln_dict = read_narayan_data('GOOGL1 raw')
MSFT1_dict,  MSFT1_ln_dict = read_narayan_data('MSFT1 raw')
TSLA1_dict,  TSLA1_ln_dict = read_narayan_data('TSLA1 raw')



nprices_dicts2 = [AAPL2_dict, AMZN2_dict, GOOGL2_dict, MSFT2_dict, TSLA2_dict]
nlnreturns_dicts2 = [AAPl2_ln_dict, AMZN2_ln_dict, GOOGL2_ln_dict, MSFT2_ln_dict, TSLA2_ln_dict]

nprices_dicts1 = [AAPL1_dict, AMZN1_dict, GOOGL1_dict, MSFT1_dict, TSLA1_dict]
nlnreturns_dicts1 = [AAPl1_ln_dict, AMZN1_ln_dict, GOOGL1_ln_dict, MSFT1_ln_dict, TSLA1_ln_dict]

for i, dict in enumerate(nprices_dicts1):
    dict.update(nprices_dicts2[i])

for i, dict in enumerate(nlnreturns_dicts1):
    dict.update(nlnreturns_dicts2[i])

SP5_dict, SP5_ln_dict = SP5_constructor(nprices_dicts1)
nprices_dicts1.append(SP5_dict)
nlnreturns_dicts1.append(SP5_ln_dict)
stocks = ['datetime', 'AAPL2 raw','AMZN2 raw','GOOGL2 raw','MSFT2 raw','TSLA2 raw', 'SP5 raw']



write_time_series_to_file('narayan_timeseries', stocks , nprices_dicts1)
write_time_series_to_file('logreturn_narayan_timeseries', stocks , nlnreturns_dicts1)



