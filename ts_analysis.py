#!/usr/bin/python

import scipy.stats as scs
import bs4 as bs
import datetime as dt
import os
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import csv
import urllib2
import scipy
import scipy.fftpack
import statsmodels.tsa.stattools as ts
import statsmodels as smt
import random
import glob
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from pylab import *
from matplotlib import style


def get_headers(df):
    return(df.columns.values)

def tsplot(data, lags=None, figsize=(10,8), style='ggplot', tick=''):
    y = data.values
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        data.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots - ' + str(tick))        
        smt.graphics.tsaplots.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.tsaplots.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
        plt.show()


def get_tickers(tickers):
    for ticker in tickers:
        ticker = str(ticker)
        try: 
            print(ticker)
            quandl.ApiConfig.api_key = "M3S6cLgQ3b_czSDmKJxD"
            df = quandl.get("WIKI/" + ticker, start_date = "2015-12-31", end_date = "2019-03-31")
            write_to(df,str(ticker),"csv")
        except ValueError:
            print("Error")
            print(ticker)

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    with open("sp500tickers.txt", 'w') as f:
        for item in tickers:
            f.write("%s\n" % item)        
    return(tickers)
    
def read_data(ticker,flag_set_index):
    filename = 'stock_dfs/' + str(ticker) + '.csv'
    df = pd.read_csv(filename,sep=',')
    df["Date"] = pd.to_datetime(df["Date"])
    if(flag_set_index == True):
        df.set_index("Date",inplace=True)
    return(df)

def view_data(df):
    print(df.head())
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)

def set_consts():
    wnd_mean4,wnd_mean3,wnd_mean2,wnd_mean1,wnd_mean0 = 252,100,50,20,10
    keys = [ "wnd_mean4", "wnd_mean3", "wnd_mean2", "wnd_mean1", "wnd_mean0" ]
    wnds = [  wnd_mean4 ,  wnd_mean3 ,  wnd_mean2 ,  wnd_mean1 ,  wnd_mean0  ]
    res = dict(zip(keys,wnds))
    return(res)

def log_ret_all(dfs,tickers):
    cols_base = dfs[0].columns.values
    cols = []
    for tck in tickers:
        for el in cols_base:
            cols.append(el + " - " + tck)
    df = merge_dfs(dfs)
    df.columns = cols
    return(df)

def merge_dfs(dfs):
    df = pd.concat(dfs,axis=1)
    return(df)

def get_ret_values(df):
    wnd = set_consts()
    df["hl_pct"] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df["return"] = df["Adj. Close"]/df["Adj. Close"].shift(1)
    df["log-return"] = np.log(df["Adj. Close"]/df["Adj. Close"].shift(1))
    df["volatility"] = df["log-return"].rolling(window=wnd["wnd_mean4"],center=False).std() * \
                                                        np.sqrt(wnd["wnd_mean4"])
    df["roll-mean" + str(wnd["wnd_mean4"])] = df["Adj. Close"].rolling \
                                            (window=wnd["wnd_mean4"],center=False).mean()
    df["roll-mean" + str(wnd["wnd_mean3"])] = df["Adj. Close"].rolling\
                                            (window=wnd["wnd_mean3"],center=False).mean()    
    df["roll-mean" + str(wnd["wnd_mean2"])] = df["Adj. Close"].rolling\
                                            (window=wnd["wnd_mean2"],center=False).mean()    
    df["roll-mean" + str(wnd["wnd_mean1"])] = df["Adj. Close"].rolling\
                                            (window=wnd["wnd_mean1"],center=False).mean()    
    df["roll-mean" + str(wnd["wnd_mean0"])] = df["Adj. Close"].rolling\
                                            (window=wnd["wnd_mean0"],center=False).mean()    
    df["rel-return"] = df["Adj. Close"].pct_change(1)
    return(df)

def ts_analysis(df,ticker,header,lag):
    select_cols = [ col for col in df.columns if str(ticker) in col ]
    select_res = [ df[el] for el in select_cols ]
    select_df = merge_dfs(select_res)
    data = select_df[str(header) + " - " + str(ticker)].dropna()
    tsplot(data,lags=lag,figsize=(10,8),style='ggplot',tick=ticker)
    plt.show()
    
def scatter_analysis(df,tickers,header):
    select_cols = list(map(lambda x: str(header) + ' - ' + x, tickers))
    select_df = df[df.columns.intersection(select_cols)].dropna()
    pd.plotting.scatter_matrix(select_df, diagonal='kde', alpha=0.1,figsize=(12,12))
    plt.show()
        
if __name__ == '__main__':
    tickers = save_sp500_tickers()
    # get_tickers(tickers)

    dirc = os.path.join(os.getcwd(),"stock_dfs") 
    fls_abs = glob.glob(os.path.join(dirc,str("*.csv")))
    fls_rel = [ os.path.split(el)[1] for el in fls_abs ]     
    tickers_eff = [ el.split(".csv")[0] for el in fls_rel ]
    
    dfs = [ read_data(el,True) for el in tickers_eff ]
    dfs_ret = [ get_ret_values(dfs[el]) for el in range(len(dfs)) ]   
    df_all = log_ret_all(dfs_ret,tickers_eff)

    ts_analysis(df_all,"AAPL","Adj. Close",30)
    scatter_analysis(df_all,["AAPL","AMZN"],"Adj. Close")
