# -*- coding: utf-8 -*-
"""


strategyName:
    BigMom
edition: 
    functions
strategyType:
    strategy_factor
Description: 
    多因子框架中的函数部分
TODOs:    

Created on Thu Sep 21 09:32:41 2023

Edited on Thu Sep 21 09:32:41 2023


@author: oOoOo_Andra
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm


### 自定义函数, make_function函数群
def SUM(df,window=10):
    return df.rolling(window).sum()

def ABS(df):
    return abs(df)

def MEAN(df,window=10):
    return df.rolling(window).mean()

def STD(df,window=10):
    return df.rolling(window).std()

def MAX(A,B):
    return np.maximum(A,B)

def MIN(A,B):
    return np.minimum(A,B)

def CORR(x,y,window=10):
    return x.rolling(window).corr(y)

def COVIANCE(x,y,window=10):
    return x.rolling(window).cov(y)

def rolling_rank(na):
    return rankdata(na)[-1]

def TSRANK(df,window=10):
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    return np.prod(na)

def PROD(df,window=10):
    return df.rolling(window).apply(rolling_prod)

def TSMIN(df,window=10):
    return df.rolling(window).min()

def TSMAX(df,window=10):
    return df.rolling(window).max()

def DELTA(df,period=1):
    return df.diff(period)

def DELAY(df,period=1):
    return df.shift(period)

def RANK(df,pct=True):
    return df.rank(axis=1,pct=pct)

def SCALE(df,k=1):
    return df.mul(k).div(np.abs(df).sum())

def SIGN(df):
    return 1*(df > 0) - 1* (df <0)

def LOG(df):
    return np.log(df)

def WMA(df,window=10):
    weights = 0.9*np.arange(window, 0, -1)
    return df.rolling(window).apply(np.average,args=(None, weights))

def DECAYLINEAR(df, window=10):
    weights = np.arange(window, 0, -1) 
    weights = weights / sum(weights)
    return df.rolling(window).apply(np.average,args=(None, weights))

def func_decaylinear(na):
    n = len(na)
    decay_weights = np.arange(1,n+1,1) 
    decay_weights = decay_weights / decay_weights.sum()


def SEQUENCE(n):
    return np.arange(n)+1

def REGBETA(df,B,n=5):
    #arr = SEQUENCE(n)    
    arr = sm.add_constant(SEQUENCE(n))    
    return df.rolling(n).apply(lambda x: sm.OLS(x,arr).fit().params[1])
 
def REGRESI(df,B,n=5):
    '''前 n 期样本 A 对 B 做回归所得的残差'''
    arr = sm.add_constant(B)    
    return df.rolling(n).apply(lambda x: sm.OLS(x,arr).fit().resid.sum())

def SUMIF(df, n, condition):
    '''对 A 前 n 项条件求和，其中 condition 表示选择条件'''
    return df * condition.rolling(n).sum()

def FILTER(df, condition):
    """对 A 筛选出符合选择条件 condition 的样本"""
    return df * condition
    
def HIGHDAY(df, window=10):
    """计算 A 前 n 期时间序列中最大值距离当前时点的间隔"""
    return df.rolling(window).apply(np.argmax)+1

def LOWDAY(df, window=10):
    return df.rolling(window).apply(np.argmin)+1

def rolling_cumsum(na):
    return np.cumsum(na,axis=0)[-1]

def SUMAC(df, window=3):
    """计算 A 的前 n 项的累加"""
    #return df.rolling(n).apply(rolling_cumsum)
    return df.rolling(window).apply(rolling_cumsum)

def COUNT(condition, n):
    """计算前 n 期满足条件 condition 的样本个数"""
    return condition.rolling(n).sum()

def IFELSE(condition, A, B):
    '''
    IFELSE(SUM((IFELSE(CLOSE>DELAY(CLOSE,1),VOLUME,0),26))
        /SUM(IFELSE(CLOSE<=DELAY(CLOSE,1),VOLUME,0)),26)*100)
    '''
    return A * condition + B * (~condition)

def SMA(df,n,m):
    return df.ewm(alpha=m/n).mean()

def VAR(df, n):
    return df.rolling(n).var()

def SKEW(df, n):
    return df.rolling(n).skew()

def KURT(df, n):
    return df.rolling(n).kurt()

def SHARPE(df, n):
    yearRet = (df.rolling(n).mean())*242
    yearStd = (df.rolling(n).std())* (252 ** 0.5)
    return yearRet / yearStd
    

