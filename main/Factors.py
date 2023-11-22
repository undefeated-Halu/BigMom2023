# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:07:21 2021

@author: Andra Wang
"""
# import os
# os.chdir(r'E:\pytho_script\multi_factor_cta\cta_MultiFactor')
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import rankdata
import scipy as sp
import  matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime 
from datetime import timedelta
import time
# import tushare as ts
# pro = ts.pro_api('c9aa112130d488a5b336cde159a2c821cfd9d09bd3bacc939edb6bee')
# from jqdatasdk import *
# auth('13472470023','Wangjun1016')
import talib as ta 
from scipy.stats.mstats import winsorize
# from sklearn import preprocessing
from scipy import stats
from scipy.stats import rankdata
import statsmodels.api as sm
import sys
sys.path.append('E:/pytho_script/multi_factor_cta/strategyScript/')
import TAFunc as ta_wj
#%% calculator func
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
    


"""
def LongShortTest(expression,start=1, end=11, step=1,groupNum=5, length=testLenDay, 
                  fig=False,name='ALPHA_X'):
    #X['n'] = range(start, end, step)
    retList = []
    ICIRList = []
    ICList = []
    SRList = []
    mean_period_returnList = []
    if fig:
        fi,ax = plt.subplots()

    for n in range(start, end, step):
        #n=5
        X['N'] = n        
        alpha = pd.DataFrame(eval(expression,X))
        alpha.index = X['CLOSE'].index
        alpha.columns=X['CLOSE'].columns
        factor_quantile = qcut_wj(alpha,groupNum=5)
        
        dfRet = pd.DataFrame()
        gList = np.arange(groupNum) +1
        
        for g in gList:
            dfRet['group_'+str(g)] = (PCT * 1 * (factor_quantile == g)).mean(axis=1)
        
        mean_period_return = dfRet.mean()
        mean_period_returnList.append(mean_period_return.values)
 
        grossRet = dfRet['group_'+str(gList[0])] - dfRet['group_'+str(gList[-1])]
        grossCumRet = grossRet.fillna(0).cumsum()            
        
        if grossCumRet[-1] < 0:
            grossRet = -grossRet
            grossCumRet = -grossCumRet
        
        grossDaily = grossRet.resample('D').sum().dropna()
        grossDailyCumRet = grossDaily.cumsum() 
        yearRet = grossDailyCumRet[-1] * (252/testLenDay) 
        retList.append(yearRet)   
        # gross 
        
        yearStd = grossDaily.std() * (252 ** 0.5)
        SR = (yearRet-0.03)/yearStd
        SRList.append(SR)
#==============================================================================
#         # net
#         fee = (abs(pick.diff()) * 1  * 1 / 100 /100).sum(axis=1)        
#         netRet = grossRet - fee
#         
#         netDaily = netRet.resample('D').sum().dropna()
#         netDailyCumRet = netDaily.cumsum()  
#         netyearRet = netDailyCumRet[-1] * (252/testLenDay) 
#         netretList.append(netyearRet)   
#         
#        netyearStd = netDaily.std() * (252 ** 0.5)
#        netSR = (netyearRet-0.03)/netyearStd
#        netSRList.append(netSR)
#==============================================================================
        
        # IC IR
        IC = PCT.corrwith(factor_quantile,axis=1).fillna(0)
        
        ICList.append(IC.mean())
        try:
            ICIR = IC.mean() / IC.std()        
            ICIRList.append(ICIR)
        except:
            ICIRList.append(0)
                
        if fig:
            ax.plot(grossCumRet,label='n='+str(n),alpha=0.8)
    
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig(name+".png")

    res = pd.DataFrame(list(zip(range(start, end, step), retList, SRList,ICList,ICIRList)),
                        columns=['PARAM_N','grossRet','SR','IC','ICIR']) 
    res = pd.concat([res,pd.DataFrame(mean_period_returnList,columns = dfRet.columns)],axis=1)
    return res
"""
### 
def Pattern(df):
    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
    dfpattern = pd.DataFrame({
        'Y': df.Close.shift(-1) / df.Close - 1,
        'TwoCrows': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # D
        'ThreeBlackCrows': ta.CDL3BLACKCROWS(df.Open, df.High, df.Low, df.Close),  # D
        'ThreeInsideUD': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # U
        'ThreeLineStrike': ta.CDL3LINESTRIKE(df.Open, df.High, df.Low, df.Close),  # D
        'ThreeOutsideUD': ta.CDL3OUTSIDE(df.Open, df.High, df.Low, df.Close),  # U
        'ThreeStarsInTheSouth': ta.CDL3STARSINSOUTH(df.Open, df.High, df.Low, df.Close),  # U
        'ThreeAdvancingWhiteSoldiers': ta.CDL3WHITESOLDIERS(df.Open, df.High, df.Low, df.Close),  # U
        'AdvanceBlock': ta.CDLADVANCEBLOCK(df.Open, df.High, df.Low, df.lose),  # U
        'AbandonedBaby': ta.CDLABANDONEDBABY(df.Open, df.High, df.Low, df.Close, penetration=0),  # R
        'BeltHold': ta.CDLBELTHOLD(df.Open, df.High, df.Low, df.Close),  # U
        'Breakaway': ta.CDLBREAKAWAY(df.Open, df.High, df.Low, df.Close),  # U
        'ClosingMarubozu': ta.CDLCLOSINGMARUBOZU(df.Open, df.High, df.Low, df.Close),  # M
        'ConcealingBabySwallow': ta.CDLCONCEALBABYSWALL(df.Open, df.High, df.Low, df.Close),  # U
        'Counterattack': ta.CDLCOUNTERATTACK(df.Open, df.High, df.Low, df.Close),  #
        'DarkCloudCover': ta.CDLDARKCLOUDCOVER(df.Open, df.High, df.Low, df.Close, penetration=0),  # D
        'Doji': ta.CDLDOJI(df.Open, df.High, df.Low, df.Close),  #
        'DojiStar': ta.CDLDOJISTAR(df.Open, df.High, df.Low, df.Close),  # R
        'DragonflyDoji': ta.CDLDRAGONFLYDOJI(df.Open, df.High, df.Low, df.Close),  # R
        'EngulfingPattern': ta.CDLENGULFING(df.Open, df.High, df.Low, df.Close),  # R
        'EveningDojiStar': ta.CDLEVENINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RD
        'EveningStar': ta.CDLEVENINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
        'UDgapSideWhiteLines': ta.CDLGAPSIDESIDEWHITE(df.Open, df.High, df.Low, df.Close),  # M
        'GravestoneDoji': ta.CDLGRAVESTONEDOJI(df.Open, df.High, df.Low, df.Close),  # RU
        'Hammer': ta.CDLHAMMER(df.Open, df.High, df.Low, df.Close),  # R
        'HangingMan': ta.CDLHANGINGMAN(df.Open, df.High, df.Low, df.Close),  # R
        'HaramiPattern': ta.CDLHARAMI(df.Open, df.High, df.Low, df.Close),  # RU
        'HaramiCrossPattern': ta.CDLHARAMICROSS(df.Open, df.High, df.Low, df.Close),  # R
        'HighWaveCandle': ta.CDLHIGHWAVE(df.Open, df.High, df.Low, df.Close),  # R
        'HikkakePattern': ta.CDLHIKKAKE(df.Open, df.High, df.Low, df.Close),  # R
        'ModifiedHikkakePattern': ta.CDLHIKKAKEMOD(df.Open, df.High, df.Low, df.Close),  # M
        'HomingPigeon': ta.CDLHOMINGPIGEON(df.Open, df.High, df.Low, df.Close),  # R
        'IdenticalThreeCrow': ta.CDLIDENTICAL3CROWS(df.Open, df.High, df.Low, df.Close),  # D
        'InNeckPattern': ta.CDLINNECK(df.Open, df.High, df.Low, df.Close),  # D
        'InvertedHammer': ta.CDLINVERTEDHAMMER(df.Open, df.High, df.Low, df.Close),  # R
        'Kicking': ta.CDLKICKING(df.Open, df.High, df.Low, df.Close),  #
        'KickingByLength': ta.CDLKICKINGBYLENGTH(df.Open, df.High, df.Low, df.Close),  #
        'LadderBottom': ta.CDLLADDERBOTTOM(df.Open, df.High, df.Low, df.Close),  # RU
        'LongLeggedDoji': ta.CDLLONGLEGGEDDOJI(df.Open, df.High, df.Low, df.Close),  #
        'LongLineCandle': ta.CDLLONGLINE(df.Open, df.High, df.Low, df.Close),  #
        'Marubozu': ta.CDLMARUBOZU(df.Open, df.High, df.Low, df.Close),  #
        'MatchingLow': ta.CDLMATCHINGLOW(df.Open, df.High, df.Low, df.Close),  #
        'MatHold': ta.CDLMATHOLD(df.Open, df.High, df.Low, df.Close, penetration=0),  # M
        'MorningDoji': ta.CDLMORNINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
        'MorningStar': ta.CDLMORNINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
        'OnNeckPattern': ta.CDLONNECK(df.Open, df.High, df.Low, df.Close),  # MD
        'PiercingPattern': ta.CDLPIERCING(df.Open, df.High, df.Low, df.Close),  # RU
        'RickshawMan': ta.CDLRICKSHAWMAN(df.Open, df.High, df.Low, df.Close),  #
        'RFThreeMethods': ta.CDLRISEFALL3METHODS(df.Open, df.High, df.Low, df.Close),  # U
        'SeparatingLines': ta.CDLSEPARATINGLINES(df.Open, df.High, df.Low, df.Close),  # M
        'ShootingStar': ta.CDLSHOOTINGSTAR(df.Open, df.High, df.Low, df.Close),  # D
        'ShortLineCandle': ta.CDLSHORTLINE(df.Open, df.High, df.Low, df.Close),  #
        'SpinningTop': ta.CDLSPINNINGTOP(df.Open, df.High, df.Low, df.Close),  #
        'StalledPattern': ta.CDLSTALLEDPATTERN(df.Open, df.High, df.Low, df.Close),  # EU
        'StickSandwich': ta.CDLSTICKSANDWICH(df.Open, df.High, df.Low, df.Close),  #
        'Takuri': ta.CDLTAKURI(df.Open, df.High, df.Low, df.Close),  #
        'TasukiGap': ta.CDLTASUKIGAP(df.Open, df.High, df.Low, df.Close),  # MU
        'ThrustingPattern': ta.CDLTHRUSTING(df.Open, df.High, df.Low, df.Close),  # M
        'TristarPattern': ta.CDLTRISTAR(df.Open, df.High, df.Low, df.Close),  # R
        'UniqueRiver': ta.CDLUNIQUE3RIVER(df.Open, df.High, df.Low, df.Close),  # R
        'UGapTwoCrows': ta.CDLUPSIDEGAP2CROWS(df.Open, df.High, df.Low, df.Close),  # U
        'UDGapThreeMethods': ta.CDLXSIDEGAP3METHODS(df.Open, df.High, df.Low, df.Close)  # U
    })
    return dfpattern
#%% main logic func

def normalize(factor):
    '''正态化'''
    mean = factor.mean()
    std = factor.std()
    return (factor - mean) / std
    
def mad3(factor, k=3):
    '''3倍std去极值'''
    def func(ft):
        mean = ft.mean()
        std = k * ft.std() 
        up = mean + std
        dn = mean - std
        ft[ft > up] = up
        ft[ft < dn] = dn
        return ft
    res = factor.apply(func,axis=1)
    return res
    

def get_returns(symboltype):
    dataset = pd.read_csv(r'E:\pytho_script\bskMonitor\future_dataset\indexFutureClose.csv', 
                          index_col=0, parse_dates=True)
    col = dataset.columns.tolist()
    
    dataset.columns = [c[:-9] for c in col]
    
    df = dataset[symboltype]
    return df.pct_change().fillna(0)


def load_data(symbol, path, start_date='2016', end_date='2021-06-30'):
    field = ['open','close','low','high','avg','pre_close','volume']
    
    filepath = f'''{path}{symbol}_day.csv'''
    
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)[start_date:end_date].drop_duplicates()
    
    data = data[field]
    
    data.rename(columns={"avg": "avg_price"},inplace=True)

    data['amount'] = data.avg_price * data.volume
    
    return data

def load_dataset(symbol_list, path, start_date='2016', end_date='2021-06-30'):
    dataset = pd.DataFrame()
    for symbol in symbol_list:
        data = load_data(symbol, path=path,start_date=start_date, end_date=end_date)
        data['symbol'] = symbol
        data.set_index('symbol',append=True, inplace=True)
        dataset = pd.concat([dataset,data])
    
    dataset = dataset.unstack(1).stack(0)    
    return dataset

def move_normalize(df_factor, m=100, mod='std_dev'):
    """
    Parameters
    ----------
    df_factor : pd.DataFrame
        factors.
    mod : str, optional
        The default is 'std_dev'.
        std_dev: 标准差正态化
        min_max: 极值化

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """
    if mod == 'std_dev':
        return df_factor.rolling(m).apply(lambda x:(x[-1] - x.mean()) / x.std())
    if mod == 'min_max':
        return df_factor.rolling(m).apply(lambda x:(x[-1] - x.min()) / (x.max() - x.min()))


'''
def pretreat_factor_section(df_factor,trade_date_list,neu=False):
    df_pretreat_factor =  pd.DataFrame(index=trade_date_list, columns=df_factor.columns)
    for date in trade_date_list:
        factor_se = df_factor.loc[date,:].dropna()
        factor_se = winsorize(factor_se, inclusive=True,  axis=1)       
        # if neu:
        #     factor_se = neutralize(factor_se, how=['jq_l1', 'market_cap'], date=date, axis=1)
        
        factor_se = standardlize(factor_se, inf2nan=True, axis=0)
        pretreat_factor_df.loc[date,list(factor_se.index)]=factor_se
        return df_pretreat_factor
'''   
'''
# factor = pd.DataFrame(demo.alpha_191())
# factors.reset_index(inplace=True,drop=False)
factors['symbol'] = 'A'
# factor.columns = ['date','value','symbol']
factors.set_index('symbol', append=True, inplace=True)

prices = price[['close']].copy()
prices.columns = ['A']

returns = compute_forward_returns(prices,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True)


'''

def compute_forward_returns(prices,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True):
    df = pd.DataFrame()    
    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan
        
        label = str(period)+'D'
        
        forward_returns['cycle'] = label
        forward_returns.set_index('cycle',append=True, inplace=True)
        df = pd.concat([df, forward_returns])
    return df


def compute_forward_returns_complex(factor,
                                    prices,
                                    periods=(1, 5, 10),
                                    filter_zscore=None,
                                    cumulative_returns=True):
    """
    计算未来 N 期的（累计）收益率
    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex  的2维序列，
        时间 (level 0) and 标的名称(level 1)
        值为因子值
    prices : pd.DataFrame
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
    """

    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise ValueError("The timezone of 'factor' is not the "
                        "same as the timezone of 'prices'. See "
                        "the pandas methods tz_localize and "
                        "tz_convert.")

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices 时间不匹配")

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    # column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = \
            returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the correct period length as
        # there could be non-trading days which would make the computation
        # wrong if made only one test
        #
        label = str(period)+'D'
        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'symbol']),inplace=True)
    df = df.reindex(factor.index)
    return df


def qcut_wj(alpha,groupNum=5):
    rankA = alpha.rank(axis=1)-1
    mulit = alpha.shape[1] / groupNum
    factor_quantile = pd.DataFrame(0,index=alpha.index, columns=alpha.columns)    
    st = 0
    for g in np.arange(groupNum) +1:
        temp = g * ((rankA < (mulit * g)) & (rankA >= (mulit * st)) )
        factor_quantile = factor_quantile + temp
        st +=1
    return factor_quantile

def calc_cumulative(df_ic,by_group_date=False):
    if by_group_date:        
        grouper = df_ic.index.get_level_values('cycle').to_frame()
        grouper = grouper.apply(lambda x: int(x[:-1]) for x in grouper)
        df_ic_cum = (df_ic.fillna(0)/(grouper.values)).groupby(level=2).cumsum()
    else:
        df_ic_cum = df_ic.fillna(0).groupby(level=1).cumsum()        
    
    return df_ic_cum

def calc_normal_ic_ts(factor, ret, n=100, by_group_date=False):
    '''
    单品种 时序上的 IC 值
    Parameters
    ----------
    factors : pd.DataFrame
        MultiIndex: 'Index', 'symbol'
    ret : pd.DataFrame
        MultiIndex:'Index', 'symbol'
    n : int, optional
        时序上算IC的滚动窗口. The default is 100.

    Returns
    -------
    df_ic : pd.DataFrame
        MultiIndex
        Example:
        index      symbol cycle  ||  Alpha1  Alpha2  Alpha3  Alpha4  Alpha5  Alpha191                                              
        2020-01-02   A      1D   ||   NaN     NaN     NaN     NaN     NaN       NaN
        2020-01-03   A      5D   ||   NaN     NaN     NaN     NaN     NaN       NaN
    '''
    df_ic = pd.DataFrame()
    for i in range(ret.shape[1]):
        # ic_list.append(factors.corrwith(ret.iloc[:,i]))
        temp = factor.rolling(n).corr(ret.iloc[:,i])
        temp['cycle'] = ret.columns[i]
        temp.set_index('cycle',append=True, inplace=True)
        df_ic = pd.concat([df_ic, temp])
        
    df_ic.sort_index(level='cycle',ascending=True,inplace=True)
    
    if by_group_date:        
        grouper = df_ic.index.get_level_values('cycle').to_frame()
        grouper = grouper.apply(lambda x: int(x[:-1]) for x in grouper)
        df_ic_cum = (df_ic.fillna(0)/(grouper.values)).groupby(level=1).cumsum()
    else:
        df_ic_cum = df_ic.fillna(0).groupby(level=1).cumsum()        
    
    return df_ic_cum    
# df_ic.to_csv('demo_ic_multiCycle.csv')

def calc_normal_ic_ts_multi(factor, ret, n=60, mode='mean'):
    '''
    单因子 单期收益率 多品种 时序上的 IC 值
    用于测试因子在整体品种的解释力

    Parameters
    ----------
    factor : pd.DataFrame
        index = datetime
        columns = 品种list
    ret : pd.DataFrame
        index = datetime
        columns = 品种list
    n : int, optional
        相关性样本长度 The default is 30.
    
    mode : str, optional
        统计方式: mean 或者 median

    Returns
    -------
    None.

    '''
    if factor.shape[0] != ret.shape[0]:
        raise KeyError("Time length not in mapping")
    elif factor.shape[1] != ret.shape[1]:
        raise KeyError("Assets not in mapping")
    else:
        ret = ret[factor.columns]
    
    ic = factor.rolling(n).corr(ret).replace([float('-inf'),float('inf')],0)
    if mode == 'mean':
        ic_avg_cum = ic.mean(axis=1).fillna(0).cumsum()
    if mode == 'median':
        ic_avg_cum = ic.median(axis=1).fillna(0).cumsum()
        
    return ic, ic_avg_cum


def calc_normal_ic_sec(factor, ret, method='normal'):   
    """
    Parameters
    ----------
    factor : pd.DataFrame
        MultiIndex:'date', 'factor'
        columns = 品种名称
    ret : pd.DataFrame
        columns = 品种名称
    TYPE : str, optional
        'normal'
        'rank'
         The default is 'normal'.
    by_group : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    if method == 'rank':
        factor = factor.rank(method='average',axis=1)
        
    list_factor = factor.index.unique(level=1).to_list()
    result = pd.DataFrame()
    
    for f in list_factor:
        # print(f)
        dff = factor.loc[(slice(None),f),:].reset_index(level=1,drop=True)
        res = dff.corrwith(ret.reset_index(level=1,drop=True),axis=1)
        result = pd.concat([result, res],axis=1)       
    result.columns = [list_factor]
    return result

def calc_normal_ic_sec_group(factor_group, return_factor_cycle_group):  
    if len(factor_group.columns) != len(return_factor_cycle_group.columns):
        raise KeyError("Returns not in mapping")
    else:
        factor_group.sort_index(level='factor',ascending=True,inplace=True)
        ret = return_factor_cycle_group.droplevel(1)
        ret.sort_index(level='factor',ascending=True,inplace=True)
        result =  factor_group.corrwith(ret,axis=1).to_frame()
        result.columns = return_factor_cycle_group.index.unique(level=1).to_list()
    return result

# TODO:
    # duplicates='drop'
def grouper_factor(factor, groupNum=5, method='mean'):
    
    factor_group_rank = (factor.rank(method='average',axis=1)).apply(lambda x
                                                                : pd.cut(x,bins=groupNum, labels=False,duplicates='drop'),axis=1) + 1
    factor_group = pd.DataFrame()
    for i in range(1,groupNum+1):
        if method == 'mean':
            temp = ((factor_group_rank[factor_group_rank == i]) * factor).mean(axis=1).to_frame()
        if method == 'median':
            temp = ((factor_group_rank[factor_group_rank == i]) * factor).median(axis=1).to_frame()
            
        factor_group = pd.concat([factor_group,temp],axis=1)
    factor_group.columns = [('group_' + str(i)) for i in range(1, groupNum+1)]
    return factor_group_rank, factor_group

def grouper_return(factor_group_rank, ret, groupNum=3):
    list_factor = factor_group_rank.index.unique(level=1).to_list()
    # 对其标的轴
    if len(factor_group_rank.columns) != len(ret.columns):
        raise KeyError("Assets not in mapping")
    else:
        ret = ret[factor_group_rank.columns]
    
    factor_return_cycle_group = pd.DataFrame()
    for i in range(1,groupNum+1):
        temp = (factor_group_rank[factor_group_rank == i])
        factor_return_group = pd.DataFrame()
        for f in list_factor:
            dff = temp.loc[(slice(None),f),:].reset_index(level=1,drop=True)
            res = pd.DataFrame((ret.values) * (dff.values) / i,  index=ret.index, columns=dff.columns).mean(axis=1).to_frame()
            res.loc[:,'factor'] = f
            res.set_index('factor',append=True, inplace=True)
            factor_return_group = pd.concat([factor_return_group,res],axis=0)
        
        factor_return_cycle_group = pd.concat([factor_return_cycle_group,factor_return_group],axis=1)
        
    factor_return_cycle_group.columns = [('group_' + str(i)) for i in range(1, groupNum+1)]
    return factor_return_cycle_group
    
def calc_icir(ic,n=60):
    icir = ic.groupby('cycle').rolling(n).apply(lambda x: x.mean() / x.std())
    return icir    

# def winsorize_wj(factor):
#     factor = pd.DataFrame(winsorize(factor,scale=3),columns=factor.columns,
#                           index=factor.index)

#     pass

#%% factor
'''
        OPEN       = price.loc[(slice(None),'open'),:].droplevel(1)[symbol_list]
        CLOSE      = price.loc[(slice(None),'close'),:].droplevel(1)[symbol_list]
        LOW        = price.loc[(slice(None),'low'),:].droplevel(1)[symbol_list]
        HIGH       = price.loc[(slice(None),'high'),:].droplevel(1)[symbol_list]
        VWAP       = price.loc[(slice(None),'avg_price'),:].droplevel(1)[symbol_list]
        PRE_CLOSE  = price.loc[(slice(None),'pre_close'),:].droplevel(1)[symbol_list]
        VOLUME     = price.loc[(slice(None),'volume'),:].droplevel(1)[symbol_list]
        AMOUNT     = price.loc[(slice(None),'amount'),:].droplevel(1)[symbol_list]
        RET        = price.loc[(slice(None),'close'),:].droplevel(1).pct_change().fillna(0)[symbol_list]
'''
class WTS_factor:
    def __init__(self, symbol_list, path, start_date, end_date):
        
        price = load_dataset(symbol_list, path, start_date, end_date)[symbol_list]
        # benchmark_price = get_price(index, None, end_date, '1d',['open','close','low','high','avg_price','prev_close','volume'], False, None, 250,is_panel=1)
        self.OPEN       = price.loc[(slice(None),'open'),:].droplevel(1)[symbol_list]
        self.CLOSE      = price.loc[(slice(None),'close'),:].droplevel(1)[symbol_list]
        self.LOW        = price.loc[(slice(None),'low'),:].droplevel(1)[symbol_list]
        self.HIGH       = price.loc[(slice(None),'high'),:].droplevel(1)[symbol_list]
        self.VWAP       = price.loc[(slice(None),'avg_price'),:].droplevel(1)[symbol_list]
        self.PRE_CLOSE  = price.loc[(slice(None),'pre_close'),:].droplevel(1)[symbol_list]
        self.VOLUME     = price.loc[(slice(None),'volume'),:].droplevel(1)[symbol_list]
        self.AMOUNT     = price.loc[(slice(None),'amount'),:].droplevel(1)[symbol_list]
        self.RET        = price.loc[(slice(None),'close'),:].droplevel(1).pct_change().fillna(0)[symbol_list]
        self.HD         = self.HIGH - DELAY(self.HIGH,1)
        self.LD         = DELAY(self.LOW,1) - self.LOW 
        self.TR         = MAX(MAX(self.HIGH-self.LOW, ABS(self.HIGH-DELAY(self.CLOSE,1))),ABS(self.LOW-DELAY(self.CLOSE,1)))
        # self.benchmark_open_price = benchmark_price.loc[:, 'open']
        # self.benchmark_close_price = benchmark_price.loc[:, 'close']
        
        
    def alpha_001(self,N=6):
        # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
        return (-1 * CORR(RANK(DELTA(LOG(self.VOLUME), 1)), RANK(((self.CLOSE - self.OPEN) / self.OPEN)), N))
    
    def alpha_002(self):
        # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1)) 
        return -1 * DELTA(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW), 1)


    def alpha_003(self,N=6):
        # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
        return SUM(IFELSE(self.CLOSE==DELAY(self.CLOSE,1),0,
                          self.CLOSE - IFELSE(self.CLOSE>DELAY(self.CLOSE,1),
                                               MIN(self.LOW,DELAY(self.CLOSE,1)),
                                               MAX(self.HIGH,DELAY(self.CLOSE,1)))),N)

    def alpha_004(self,N=8,M=2):
        '''
        ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) :
            (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : 
                (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        '''
        return    IFELSE((((SUM(self.CLOSE, N) / N) + STD(self.CLOSE, N)) < (SUM(self.CLOSE, M) / M)) , -1 ,
                         IFELSE(((SUM(self.CLOSE, M) / M) < ((SUM(self.CLOSE, N) / N) - STD(self.CLOSE, N))) , 1 , 
                                IFELSE(((1 < (self.VOLUME / MEAN(self.VOLUME,20))) | ((self.VOLUME /MEAN(self.VOLUME,20)) == 1)), 1 , -1 )))
        
    ## BUG
    def alpha_005(self,n=5):
        # (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
        return -1 * TSMAX(CORR(TSRANK(self.VOLUME, n), TSRANK(self.HIGH, n), n), int(np.ceil(n/2)))
    
    

    def alpha_006(self, p = 0.15):
        # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
        return RANK(SIGN(DELTA((self.OPEN * 0.85) + (self.HIGH * 0.15),4)))*-1


    def alpha_007(self, n=3):
        # ((RANK(MAX((avg_price - close), n)) + RANK(MIN((avg_price - close), n))) * RANK(DELTA(volume, n)))
        return ((RANK(MAX(self.VWAP - self.CLOSE, n)) 
                      + RANK(MIN((self.VWAP - self.CLOSE), n))) * RANK(DELTA(self.VOLUME, n)))


    def alpha_008(self, P=0.2):
        # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
        return RANK(DELTA(((((self.HIGH + self.LOW) / 2) * P) + (self.VWAP * (1-P))), 4) * -1)


    def alpha_009(self,n=7):
        # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
        return SMA(((self.HIGH+self.LOW) / 2 - (DELAY(self.HIGH,1) + DELAY(self.LOW,1)) / 2)
                   *(self.HIGH-self.LOW) / self.VOLUME,n,2)


    def alpha_010(self,N=20):
        # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
        return RANK(MAX(IFELSE(self.RET < 0, STD(self.RET, N), self.CLOSE)**2, 5))


    def alpha_011(self,N=6):
        # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
        return SUM(((self.CLOSE-self.LOW)-(self.HIGH-self.CLOSE))/(self.HIGH-self.LOW)*self.VOLUME,N)


    def alpha_012(self,N=10):
        # (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
        return (RANK((self.OPEN - (SUM(self.VWAP, N) / N)))) * (-1 * (RANK(ABS((self.CLOSE - self.VWAP)))))


    def alpha_013(self):
        # (((HIGH * LOW)**0.5) - VWAP)
        return (((self.HIGH * self.LOW)**0.5) - self.VWAP)


    def alpha_014(self,N=5):
        # CLOSE-DELAY(CLOSE,5)
        return self.CLOSE - DELAY(self.CLOSE,N)


    def alpha_015(self):
        # OPEN/DELAY(CLOSE,1)-1
        return self.OPEN/DELAY(self.CLOSE,1) - 1


    def alpha_016(self,N=5):
        # (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        return (-1 * TSMAX(RANK(CORR(RANK(self.VOLUME), RANK(self.VWAP), N)), N))


    def alpha_017(self,N=15):
        # RANK((VWAP - MAX(VWAP, 15)))**DELTA(CLOSE, 5)
        return RANK((self.VWAP - MAX(self.VWAP, N)))**DELTA(self.CLOSE, 5)


    def alpha_018(self,N=5):
        ## CLOSE/DELAY(CLOSE,5)
        return self.CLOSE/DELAY(self.CLOSE,N)


    def alpha_019(self,N=5):
        #(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
        return (IFELSE(self.CLOSE < DELAY(self.CLOSE,N),
                       (self.CLOSE - DELAY(self.CLOSE,N))/DELAY(self.CLOSE,N),
                       IFELSE(self.CLOSE==DELAY(self.CLOSE,N),0,(self.CLOSE-DELAY(self.CLOSE,N))/self.CLOSE)))


    def alpha_020(self,N=6):
        # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        return (self.CLOSE-DELAY(self.CLOSE,N))/DELAY(self.CLOSE,N)*100



    def alpha_021(self,N=6):
        # REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
        return REGBETA(MEAN(self.CLOSE,N),SEQUENCE(N))


    def alpha_022(self,N=12):
        # MEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
        M = int(N / 2)
        P = int(N / 4)
        temp1 = (self.CLOSE - MEAN(self.CLOSE,M)) / MEAN(self.CLOSE,M)
        temp2 = DELAY(temp1, P)
        return SMA(temp1 - temp2, N, 1)
    

    def alpha_023(self,N=20):
        # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+ SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
        temp1 = IFELSE(self.RET > 0 , STD(self.CLOSE,N), 0)
        temp2 = SMA(temp1, N, 1)
        temp3 = SMA(IFELSE(self.RET <= 0 , STD(self.CLOSE,N), 0), N, 1)
        
        return temp2 / (temp2 + temp3) * 100


    def alpha_024(self,N=5):
        # SMA(CLOSE-DELAY(CLOSE,5),5,1)
        return SMA((self.CLOSE/DELAY(self.CLOSE,N) - 1),N,1)


    def alpha_025(self, N=20,M=7,P=9):
        # ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 +RANK(SUM(VWAP, 250))))
        return ((-1 * RANK((DELTA(self.CLOSE, M) 
                            * (1 - RANK(DECAYLINEAR((self.VOLUME / MEAN(self.VOLUME, N)), P)))))) * 
                (1 + RANK(SUM(self.VWAP, 250))))


    def alpha_026(self,N=7,M=5):
        # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
        return ((((SUM(self.CLOSE,N) / N) / self.CLOSE) - 1) + ((CORR(self.VWAP, DELAY(self.CLOSE, M), 230))))


    def alpha_027(self,N=3):
        # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
        temp1 = DELAY(self.CLOSE,N)
        temp2 = DELAY(self.CLOSE,2*N)
        return WMA((self.CLOSE-temp1)/temp1*100
                   +(self.CLOSE-temp2)/temp2*100,4*N)


    def alpha_028(self,N=9,M=3):
        # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
        temp1 = TSMIN(self.LOW,N)
        return (3*SMA((self.CLOSE-temp1)/(TSMAX(self.HIGH,N)-temp1)*100,M,1)
                -2*SMA(SMA((self.CLOSE-temp1)/(MAX(self.HIGH,N)-TSMAX(self.LOW,N))*100,M,1),M,1))


    def alpha_029(self,N=6):
        # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        return (self.CLOSE-DELAY(self.CLOSE,N))/DELAY(self.CLOSE,N)*self.VOLUME


    def alpha_030(self):
        return 0


    def alpha_031(self, N=12):
        # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
        temp = -MEAN(self.CLOSE,N)
        return (self.CLOSE - temp) / temp * 100


    def alpha_032(self,N=3):
        # (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        return (-1 * SUM(RANK(CORR(RANK(self.HIGH), RANK(self.VOLUME), N)), N))


    def alpha_033(self, N=5):
        # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))
        temp = TSMIN(self.LOW, N)      
        return ((((-1 * temp) + DELAY(temp, N)) 
                 * RANK(((SUM(self.VWAP, 240) - SUM(self.VWAP, N*4)) / 220)))
                *TSRANK(self.VOLUME, N))


    def alpha_034(self,N=12):
        # MEAN(CLOSE,12)/CLOSE
        return MEAN(self.CLOSE,N)/self.CLOSE


    def alpha_035(self,N=15,M=7,P=0.65):
        # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)
        return (MIN(RANK(DECAYLINEAR(DELTA(self.OPEN, 1), N)),
                    RANK(DECAYLINEAR(CORR((self.VOLUME), ((self.OPEN * P) +(self.OPEN *(1-P))), 17),M))) * -1)


    def alpha_036(self,N=6):
        # RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
        return RANK(SUM(CORR(RANK(self.VOLUME), RANK(self.VWAP), N), 2))


    def alpha_037(self, N=5):
        # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
        return (-1 * RANK(((SUM(self.OPEN, N) * SUM(self.VWAP, N)) 
                           - DELAY((SUM(self.OPEN, N) * SUM(self.VWAP, N)), N*2))))


    def alpha_038(self,N=20):
        # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        return IFELSE((SUM(self.HIGH, N) / N < self.HIGH), (-1 * DELTA(self.HIGH, 2)), 0)


    def alpha_039(self,N=8,M=12,P=0.3):
        # ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
        return ((RANK(DECAYLINEAR(DELTA((self.CLOSE), 2),N)) 
                 - RANK(DECAYLINEAR(CORR(((self.VWAP * P) + (self.OPEN * (1-P))),
                                         SUM(MEAN(self.VOLUME,180), 37), 14), M))) * -1)


    def alpha_040(self,N=26):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
        return (SUM(IFELSE(self.RET > 0 ,self.VOLUME, 0),N)
                / SUM(IFELSE(self.RET <= 0,self.VOLUME, 0),N)*100)
    


    def alpha_041(self,N=5,M=3):
        # (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
        return (RANK(MAX(DELTA((self.VWAP), M), N))* -1)


    def alpha_042(self, N=10):
        # ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
        return ((-1 * RANK(STD(self.HIGH, N))) * CORR(self.HIGH, self.VOLUME, N))


    def alpha_043(self,N=6):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
        # SUM(IFELSE(RET>0, VOLUME, IFELSE(RET<=0,-VOLUME,0)),6)
        return SUM(IFELSE(self.RET>0, self.VOLUME, IFELSE(self.RET<=0,-self.VOLUME,0)),N)


    def alpha_044(self,N=6,M=10):
        # (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))
        return (TSRANK(DECAYLINEAR(CORR(((self.LOW )), MEAN(self.VOLUME,10), 7), N),4) 
                + TSRANK(DECAYLINEAR(DELTA((self.VWAP),3), M), 15)  )


    def alpha_045(self,N=15, P=0.6):
        # (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
        return ( RANK(DELTA((((self.CLOSE * P) + (self.OPEN *(1-P)))), 1)) 
                * RANK(CORR(self.VWAP, MEAN(self.VOLUME,150), N))   )


    def alpha_046(self, N=3):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
        return (MEAN(self.CLOSE,N)+MEAN(self.CLOSE,2*N)+MEAN(self.CLOSE,4*N)+MEAN(self.CLOSE,6*N))/(4*self.CLOSE)


    def alpha_047(self, N=6):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
        return SMA((TSMAX(self.HIGH,N)-self.CLOSE)/(TSMAX(self.HIGH,N)-TSMIN(self.LOW,N))*100,9,1)   


    def alpha_048(self,N=5):
        #(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) 
        # * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
        return (-1*((RANK(((SIGN((self.CLOSE - DELAY(self.CLOSE, 1))) 
                            + SIGN((DELAY(self.CLOSE, 1) - DELAY(self.CLOSE, 2)))) 
                            + SIGN((DELAY(self.CLOSE, 2) - DELAY(self.CLOSE, 3))))))
                    * SUM(self.VOLUME, N)) / SUM(self.VOLUME, 4*N)   )


    def alpha_049(self,N=12):
        # (SUM(IFELSE((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        #   /(SUM(IFELSE((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        #     +SUM(IFELSE((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)))
        temp1 = self.HIGH + self.LOW
        temp2 = DELAY(self.HIGH,1) + DELAY(self.LOW,1)
        temp3 = MAX(ABS(self.HIGH-DELAY(self.HIGH,1)),ABS(self.LOW-DELAY(self.LOW,1)))
        temp4 = SUM(IFELSE(temp1 > temp2, 0, temp3),N)
                                      
        return (temp4 / (temp4 + SUM(IFELSE(temp1 < temp2, 0, temp3),N)) )


    def alpha_050(self):
       # ( SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
       #  /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
       #    +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
       #  -SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
       #  /(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
       #   +SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)) )
        left = self.HIGH + self.LOW
        right = DELAY(self.HIGH,1) + DELAY(self.LOW,1)
        temp1 = left <= right
        temp2 = left >= right
        res = MAX(ABS(self.HIGH-DELAY(self.HIGH,1)),ABS(self.LOW-DELAY(self.LOW,1)))
        temp3 = SUM(IFELSE(temp1, 0 , res),12)
        temp4 = SUM(IFELSE(temp2, 0 , res),12)
        return (temp3 - temp4) / (temp3 + temp4) 


    def alpha_051(self, N=12):
    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    # /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    #   +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        left = self.HIGH + self.LOW
        right = DELAY(self.HIGH,1) + DELAY(self.LOW,1)
        temp1 = left <= right
        temp2 = left >= right
        res = MAX(ABS(self.HIGH - DELAY(self.HIGH,1)), ABS(self.LOW - DELAY(self.LOW,1)))
        temp3 = SUM(IFELSE(temp1, 0 , res),N)
        temp4 = SUM(IFELSE(temp2, 0 , res),N)
        return temp3 / (temp3 + temp4)

 
    def alpha_052(self, N=26):
        # SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
        temp = DELAY((self.HIGH+self.LOW+self.CLOSE)/3,1)
        return (SUM(MAX(0, self.HIGH - temp), N) / SUM(MAX(0, temp - self.LOW), N))
   

    def alpha_053(self, N=12):
        # COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
        return COUNT(self.CLOSE>DELAY(self.CLOSE,1),N)/N*100


    def alpha_054(self, N=10):
        #(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
        return (-1 * RANK((STD(ABS(self.CLOSE - self.OPEN)) + (self.CLOSE - self.OPEN)) + CORR(self.CLOSE, self.OPEN,N)))

    
    def alpha_055(self, N=20):
# SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))
#     /((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))
#        ?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
#        :(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))
#          ?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
#          :ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4))) 
#     *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)          
        temp1 = 16 * (self.CLOSE + (self.CLOSE-self.OPEN) / 2 -DELAY(self.OPEN,1))
        temp2 = ABS(self.HIGH - DELAY(self.CLOSE,1))
        temp3 = ABS(self.LOW - DELAY(self.CLOSE,1))
        temp4 = ABS(self.HIGH - DELAY(self.LOW,1))
        temp5 = MAX(temp2, temp3)
        return SUM(temp1 / (IFELSE(((temp2 > temp3) & (temp2 > temp4)), 
                                   (temp2 + temp3) / 2 + temp5,
                                   IFELSE(((temp3 > temp4) & (temp3 > temp2)),
                                          (temp3 + temp2) / 2 + temp5,
                                          temp4 + temp5)))* temp5, N)


    def alpha_056(self,N=19, M=13):
        # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))**5)))
        return 1*(RANK((self.OPEN - TSMIN(self.OPEN, 12))) < RANK((RANK(CORR(SUM(((self.HIGH + self.LOW) / 2), N),
                                                                           SUM(MEAN(self.VOLUME,40), N), 13))**5)))

    def alpha_057(self,N=9):
        return SMA((self.CLOSE-TSMIN(self.LOW,N))/(TSMAX(self.HIGH,N)-TSMIN(self.LOW,N))*100,3,1)


    def alpha_058(self, N=20):
        # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        return COUNT(self.CLOSE>DELAY(self.CLOSE,1),N)/N*100


    def alpha_059(self,N=20):
        # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)    
        temp = DELAY(self.CLOSE,1)
        return (SUM(IFELSE(self.CLOSE == temp,0, self.CLOSE - IFELSE(self.CLOSE > temp,
                                                                     MIN(self.LOW,temp),
                                                                     MAX(self.HIGH,temp))),N))

    def alpha_060(self,N=20):
        # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
        return SUM(((self.CLOSE-self.LOW)-(self.HIGH-self.CLOSE))/(self.HIGH-self.LOW)*self.VOLUME,N)


    def alpha_061(self, N=12, M=17):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
        return (MAX(RANK(DECAYLINEAR(DELTA(self.VWAP, 1), N)),
                    RANK(DECAYLINEAR(RANK(CORR((self.LOW),MEAN(self.VOLUME,80), 8)), M))) * -1)


    def alpha_062(self, N=5):
        # (-1 * CORR(HIGH, RANK(VOLUME), 5))
        return (-1 * CORR(self.HIGH, RANK(self.VOLUME), N))


    def alpha_063(self, N=6):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
        return SMA(MAX(self.CLOSE-DELAY(self.CLOSE,1),0),N,1)/SMA(ABS(self.CLOSE-DELAY(self.CLOSE,1)),N,1)*100


    def alpha_064(self):
        # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
        #      RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
        return (MAX(RANK(DECAYLINEAR(CORR(RANK(self.VWAP), RANK(self.VOLUME), 4), 4)),
                    RANK(DECAYLINEAR(MAX(CORR(RANK(self.CLOSE), RANK(MEAN(self.VOLUME,60)), 4), 13), 14))) * -1)


    def alpha_065(self, N=6):
        # MEAN(CLOSE,6)/CLOSE
        return MEAN(self.CLOSE,N)/self.CLOSE


    def alpha_066(self, N=6):
        # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
        return (self.CLOSE-MEAN(self.CLOSE,N))/MEAN(self.CLOSE,N)*100


    def alpha_067(self, N=24):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
        temp = self.CLOSE-DELAY(self.CLOSE,1)
        return SMA(MAX(temp,0),N,1)/SMA(ABS(temp),N,1)*100


    def alpha_068(self, N=15):
        #SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
        return SMA(((self.HIGH+self.LOW)/2-(DELAY(self.HIGH,1)+DELAY(self.LOW,1))/2)
                   *(self.HIGH-self.LOW)/self.VOLUME,N,2)
    
    
    def alpha_069(self):          
        return 0
    
    
    def alpha_070(self,N=6):
        # STD(AMOUNT, 6)
        return STD(self.AMOUNT,N)
    

    def alpha_071(self, N=24):
        # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100 
        return (self.CLOSE-MEAN(self.CLOSE,N)) / MEAN(self.CLOSE,N)*100
    
    
    def alpha_072(self, N=6, M=15):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        return SMA((TSMAX(self.HIGH,N)-self.CLOSE)/(TSMAX(self.HIGH,N)-TSMIN(self.LOW,6))*100,M,1)
    
    
    #############################################################################
    def alpha_073(self):
        #((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5)
        #-RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
        return ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((self.CLOSE), self.VOLUME, 10), 16), 4), 5) 
                 - RANK(DECAYLINEAR(CORR(self.VWAP, MEAN(self.VOLUME,30), 4),3))) * -1)
        
        
    #############################################################################    
    def alpha_074(self,N=20,M=40,P=0.35):
        # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) 
        #  + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))) 
        return (RANK(CORR(SUM(((self.LOW * P) + (self.VWAP *(1-P))), N), SUM(MEAN(self.VOLUME,M), N), 7)) 
                + RANK(CORR(RANK(self.VWAP), RANK(self.VOLUME), 6)))
    
    
    #############################################################################

    def alpha_075(self):
        #COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50) 
        return 0

    #############################################################################
    def alpha_076(self, N=20):
        # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20) 
        return (STD(ABS((self.CLOSE/DELAY(self.CLOSE,1)-1))/self.VOLUME,N)
                / MEAN(ABS((self.CLOSE/DELAY(self.CLOSE,1)-1))/self.VOLUME,N))
    
    
    #############################################################################
    def alpha_077(self,N=40):
        # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), 
        # RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))     
        return MIN(RANK(DECAYLINEAR(((((self.HIGH + self.LOW) / 2) + self.HIGH) 
                                     - (self.VWAP + self.HIGH)), 20)),
                   RANK(DECAYLINEAR(CORR(((self.HIGH + self.LOW) / 2), 
                                         MEAN(self.VOLUME,N), 3), 6)))
    
    
    #############################################################################
    def alpha_078(self, N=12,P=0.015):
        # ((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12)) 
        temp = (self.HIGH+self.LOW+self.CLOSE)/3
        return ((temp - MEAN(temp,N)) / (P * MEAN(ABS(self.CLOSE-MEAN(temp,N)),N)))
    
    
    #############################################################################
    def alpha_079(self, N=12):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        return SMA(MAX(self.CLOSE-DELAY(self.CLOSE,1),0),N,1)/SMA(ABS(self.CLOSE-DELAY(self.CLOSE,1)),N,1)*100
    
    
    #############################################################################
    def alpha_080(self, N=5):
        #(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        return (self.VOLUME-DELAY(self.VOLUME,N))/DELAY(self.VOLUME,N)*100
    
  
    #############################################################################
    def alpha_081(self,n=21):
        # SMA(VOLUME,21,2)
        return SMA(self.VOLUME,n,2)

    
    #############################################################################
    def alpha_082(self, N=6, M=20):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
        return SMA((TSMAX(self.HIGH,N)-self.CLOSE)/(TSMAX(self.HIGH,N)-TSMIN(self.LOW,N))*100,M,1)
  

    #############################################################################
    def alpha_083(self, N=5):
        # (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
        return (-1 * RANK(COVIANCE(RANK(self.HIGH), RANK(self.VOLUME), N)))


    #############################################################################
    def alpha_084(self, N=20):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
        return SUM(IFELSE(self.CLOSE>DELAY(self.CLOSE,1),
                          self.VOLUME,
                          IFELSE(self.CLOSE<DELAY(self.CLOSE,1),-self.VOLUME,0)),N)
    
    
    #############################################################################
    def alpha_085(self, N=20,M=8):
        # (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
        return (TSRANK((self.VOLUME / MEAN(self.VOLUME,N)), N) * TSRANK((-1 * DELTA(self.CLOSE, 7)), M))
    
    
    #############################################################################
    def alpha_086(self,N=10):
        # ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :
        # (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) *
        # (CLOSE - DELAY(CLOSE, 1)))))
        temp1 = (DELAY(self.CLOSE, 2*N) - DELAY(self.CLOSE, 10)) / 10
        temp2 = (DELAY(self.CLOSE, N) - self.CLOSE) / 10
        return IFELSE(0.25 < (temp1 - temp2),
                               -1,
                               IFELSE((temp1 - temp2)<0,
                                      1,
                                      -1 * (self.CLOSE - DELAY(self.CLOSE, 1))))

    
    #############################################################################
    def alpha_087(self, N=7, M=11, P=0.9):
        # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) 
        #   + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) 
        #                         /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1) 
        return ((RANK(DECAYLINEAR(DELTA(self.VWAP, 4), N)) 
                 + TSRANK(DECAYLINEAR(((((self.LOW * P) + (self.LOW *(1-P))) - self.VWAP)
                                       / (self.OPEN - ((self.HIGH + self.LOW) / 2))), M), N)) * -1)
    
    def alpha_088(self, N=20):
        # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
        return (self.CLOSE-DELAY(self.CLOSE,20))/DELAY(self.CLOSE,20)*100
    
    
    def alpha_089(self, N=13, M=27):
        # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
        return 2 *(SMA(self.CLOSE,N,2) - SMA(self.CLOSE,M,2) - SMA(SMA(self.CLOSE,N,2) - SMA(self.CLOSE,M,2),10,2))
    
    
    def alpha_090(self, N=5):
        # ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
        return (RANK(CORR(RANK(self.VWAP), RANK(self.VOLUME), N)) * -1)
    
     
    def alpha_091(self,N=5, M=40):
        # ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)        #################      
        return ((RANK((self.CLOSE - MAX(self.CLOSE, N))
                      )*RANK(CORR((MEAN(self.VOLUME,M)), self.LOW, N))) * -1)
    
    
    def alpha_092(self,P=0.35):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        return (MAX(RANK(DECAYLINEAR(DELTA(((self.CLOSE * P) + (self.VWAP *(1-P))), 2), 3)),
                    TSRANK(DECAYLINEAR(ABS(CORR((MEAN(self.VOLUME,180)), self.CLOSE, 13)), 5), 15)) * -1)
    
    
    def alpha_093(self,N=20):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
        return SUM(IFELSE(self.OPEN>=DELAY(self.OPEN,1), 0, MAX((self.OPEN-self.LOW),(self.OPEN-DELAY(self.OPEN,1)))),N)
    
    
    def alpha_094(self,N=30):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
        return SUM(IFELSE(self.CLOSE>DELAY(self.CLOSE,1),
                          self.VOLUME, 
                          IFELSE(self.CLOSE<DELAY(self.CLOSE,1),
                                 -self.VOLUME,
                                 0)),N)
    
    def alpha_095(self,N=20):
        # STD(AMOUNT,20) 
        return STD(self.AMOUNT,20)
    
    
    def alpha_096(self,N=9):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) 
        return SMA(SMA((self.CLOSE-TSMIN(self.LOW,N))/(TSMAX(self.HIGH,N)-TSMIN(self.LOW,N))*100,3,1),3,1)
    
    
    def alpha_097(self,N=10):
        # STD(VOLUME,10)
        return STD(self.VOLUME,N)
    
    
    def alpha_098(self,N=100,P=0.05):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)|((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))
        #  ?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        return  IFELSE(((DELTA((SUM(self.CLOSE,N) / N), N)/DELAY(self.CLOSE,N)) < P) 
                       | ((DELTA((SUM(self.CLOSE,N) / N), N) / DELAY(self.CLOSE,N)) == P),
                       -1 * (self.CLOSE -  TSMIN(self.CLOSE, N)),
                       -1 * DELTA(self.CLOSE,3))

    
    def alpha_099(self,N=5):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) 
        return (-1 * RANK(COVIANCE(RANK(self.CLOSE), RANK(self.VOLUME), N)))
    
    
    def alpha_100(self,N=20):
        # STD(VOLUME,20)
        return STD(self.VOLUME,20)
    
   
    def alpha_101(self,N=15,M=37):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) 
        return ((RANK(CORR(self.CLOSE, SUM(MEAN(self.VOLUME,2*N), M), N))
                 < RANK(CORR(RANK(((self.HIGH * 0.1) + (self.VWAP * 0.9))),RANK(self.VOLUME), 11))) * -1)
    
    
    def alpha_102(self,N=6):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
        return SMA(MAX(self.VOLUME-DELAY(self.VOLUME,1),0),N,1) / SMA(ABS(self.VOLUME-DELAY(self.VOLUME,1)),N,1)*100
    
    
    def alpha_103(self,N=20):
        # ((20-LOWDAY(LOW,20))/20)*100 
        return ((N-LOWDAY(self.LOW,N))/N)*100
    
    
    def alpha_104(self,N=5):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
        return (-1 * (DELTA(CORR(self.HIGH, self.VOLUME, N), N) * RANK(STD(self.CLOSE, 4*N))))
    
    
    def alpha_105(self,N=10):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
        return (-1 * CORR(RANK(self.OPEN), RANK(self.VOLUME), N))
    
    
    def alpha_106(self,N=20):
        # CLOSE-DELAY(CLOSE,20)
        return self.CLOSE-DELAY(self.CLOSE, N)
    
    
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) 
        return (((-1 * RANK((self.OPEN - DELAY(self.HIGH, 1)))) * RANK((self.OPEN - DELAY(self.CLOSE, 1)))) * RANK((self.OPEN - DELAY(self.LOW, 1))))
    
    def alpha_108(self,N=120):
        # ((RANK((HIGH-MIN(HIGH,2)))**RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) 
        return ((RANK((self.HIGH - MIN(self.HIGH, 2)))**RANK(CORR((self.VWAP), (MEAN(self.VOLUME,N)), 6))) * -1)
    
    
    def alpha_109(self, N=10):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        return SMA(self.HIGH-self.LOW,N,2) / SMA(SMA(self.HIGH-self.LOW,N,2),N,2)
    
    
    def alpha_110(self,N=20):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
        return SUM(MAX(0,self.HIGH-DELAY(self.CLOSE,1)),N)/SUM(MAX(0,DELAY(self.CLOSE,1)-self.LOW),N)*100
    
    
    def alpha_111(self):
        # SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)       
        return (SMA(self.VOLUME * ((self.CLOSE-self.LOW)-(self.HIGH-self.CLOSE))
                   /(self.HIGH-self.LOW),11,2)-SMA(self.VOLUME*((self.CLOSE-self.LOW)-(self.HIGH-self.CLOSE))
                                                   /(self.HIGH-self.LOW),4,2))
    
                                                   
    def alpha_112(self,N=12):
        # (SUM(IFELSE(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)
        #  -SUM(IFELSE(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))
        # /(SUM(IFELSE(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)
        #   +SUM(IFELSE(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))*100 
        temp1 = self.CLOSE - DELAY(self.CLOSE,1)
        temp1 = SUM(IFELSE(temp1 > 0, temp1, 0), N)
        temp2 = SUM(IFELSE(temp1 < 0, ABS(temp1), 0), N)
        return (temp1 - temp2) / (temp1 + temp2) * 100
    
    
    def alpha_113(self,N=20, M=5):
        # (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),
        # SUM(CLOSE, 20), 2))))
        return (-1 * ((RANK((SUM(DELAY(self.CLOSE, 5), N) / N)) 
                       * CORR(self.CLOSE, self.VOLUME, 2)) 
                      * RANK(CORR(SUM(self.CLOSE, M),SUM(self.CLOSE, N), 2))))
    
    
    def alpha_114(self, N=5):
        # ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /
        # (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
        return ((RANK(DELAY(((self.HIGH - self.LOW) / (SUM(self.CLOSE, N) / N)), 2))* RANK(RANK(self.VOLUME))) 
                / (((self.HIGH - self.LOW) /(SUM(self.CLOSE, N) / N)) / (self.VWAP - self.CLOSE))) 
    
    
    def alpha_115(self,N=30):
        # (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))**RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))
 
        return (RANK(CORR(((self.HIGH * 0.9) + (self.CLOSE * 0.1)), 
                          MEAN(self.VOLUME,N), 10))**RANK(CORR(TSRANK(((self.HIGH + self.LOW) /2), 4), TSRANK(self.VOLUME, 10), 7)))

# 20210729 STOP HERE
          
    def alpha_116(self,N=20):
        # REGBETA(CLOSE,SEQUENCE,20)
        return REGBETA(self.CLOSE,SEQUENCE,N)

    
    def alpha_117(self,N=16):
        # ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
        return ((TSRANK(self.VOLUME, 2*N) * (1 - TSRANK(((self.CLOSE + self.HIGH) - self.LOW), N))) * (1 - TSRANK(self.VWAP, 2*N))) 
    
    
    def alpha_118(self,N=20):
        # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
        return SUM(self.HIGH-self.OPEN, N) / SUM(self.OPEN-self.LOW,N)*100
    
    
    def alpha_119(self,N=5,M=15):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        return (RANK(DECAYLINEAR(CORR(self.VWAP, SUM(MEAN(self.VOLUME,N), 26), 5), 7)) 
                -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(self.OPEN), RANK(MEAN(self.VOLUME,M)), 21), 9), 7), 8)))
    
    
    def alpha_120(self):
        # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
        return (RANK((self.VWAP - self.CLOSE)) / RANK((self.VWAP + self.CLOSE)))
    
 
    def alpha_121(self,N=12,M=20,K=60):
        # ((RANK((VWAP - MIN(VWAP, 12)))**TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)
        return ((RANK((self.VWAP - MIN(self.VWAP, N)))**TSRANK(CORR(TSRANK(self.VWAP, M), TSRANK(MEAN(self.VOLUME,K), 2), 18), 3)) *-1)
    
    
    def alpha_122(self,N=13):
        # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)) / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        temp = SMA(SMA(SMA(LOG(self.CLOSE),N,2),N,2),N,2)
        return (temp - DELAY(temp, 1)) / DELAY(temp, 1)
        
        
    def alpha_123(self,N=60,M=20):
        # ((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        return ((RANK(CORR(SUM(((self.HIGH + self.LOW) / 2), M), SUM(MEAN(self.VOLUME,N), M), 9)) 
                 < RANK(CORR(self.LOW, self.VOLUME,6))) * -1)
    
    
    def alpha_124(self,N=30):
        # (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        return (self.CLOSE - self.VWAP) / DECAYLINEAR(RANK(TSMAX(self.CLOSE, N)),2)
    
    
    def alpha_125(self,N=80):
        # (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        return (RANK(DECAYLINEAR(CORR((self.VWAP), MEAN(self.VOLUME, N), 17), 20)) 
                / RANK(DECAYLINEAR(DELTA(((self.CLOSE * 0.5) + (self.VWAP * 0.5)), 3), 16)))
    
    
    def alpha_126(self):
        # (CLOSE + HIGH + LOW) / 3
        return (self.CLOSE+self.HIGH+self.LOW) / 3
    
    
    def alpha_127(self):
        return 0
    
    
    def alpha_128(self,N=14):
        # 100-(100/(1+SUM(IFELSE((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)
        #           /SUM(IFELSE((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)))
        temp1 = (self.HIGH+self.LOW+self.CLOSE) / 3
        temp2 = DELAY(temp1, 1)
        return (100-(100/(1+SUM(IFELSE(temp1 > temp2, temp1*self.VOLUME, 0),N)
                         /SUM(IFELSE(temp1 < temp2, temp1*self.VOLUME, 0),N))))
    
    
    def alpha_129(self, N=12):
        # SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        return SUM(IFELSE(self.CLOSE-DELAY(self.CLOSE,1)<0,ABS(self.CLOSE-DELAY(self.CLOSE,1)),0),N)

    
    def alpha_130(self, N=40, M=10, K=3):
        # (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) 
        # / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))   
        return (RANK(DECAYLINEAR(CORR(((self.HIGH + self.LOW) / 2), MEAN(self.VOLUME, N), 9), M)) /
                RANK(DECAYLINEAR(CORR(RANK(self.VWAP), RANK(self.VOLUME), 7),K)))
    
    def alpha_131(self,N=50,M=18):
        # (RANK(DELTA(VWAP, 1))**TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
        return (RANK(DELTA(self.VWAP, 1))**TSRANK(CORR(self.CLOSE,MEAN(self.VOLUME,N), M), M))


    def alpha_132(self,N=20):
        # MEAN(AMOUNT, 20)
        return MEAN(self.AMOUNT,N)
    
    
    def alpha_133(self, N=20):
        # ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        return ((N-HIGHDAY(self.HIGH,N))/N)*100-((N-LOWDAY(self.LOW,N))/N)*100
    
    
    def alpha_134(self, N=12):
        # (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        return (self.CLOSE-DELAY(self.CLOSE,N))/DELAY(self.CLOSE,N)*self.VOLUME
    
    
    def alpha_135(self, N=20):
        # SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        return SMA(DELAY(self.CLOSE/DELAY(self.CLOSE,N),1),N,1)

    
    def alpha_136(self, N=3, M=10):
        # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        return ((-1 * RANK(DELTA(self.VWAP, N))) * CORR(self.OPEN, self.VOLUME, M))
    
    def alpha_137(self):
        # (16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))
        #  /(IFELSE(ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)),
        #           ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4,
        #           IFELSE(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)),
        #                  ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4,
        #                  ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
        #  *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))))
        DELAY1 = DELAY(self.CLOSE,1)
        DELAY2 = DELAY(self.LOW,1)
        temp1 = ABS(self.HIGH - DELAY1)
        temp2 = ABS(self.LOW - DELAY1) 
        temp3 = ABS(self.HIGH - DELAY2) 
        temp5 = ABS(DELAY1 - DELAY(self.OPEN,1))/4
        temp4 = (temp1 + temp2) / 2 + temp5
        return (16 * (self.CLOSE - DELAY1 + (self.CLOSE - self.OPEN)/2 + DELAY1 - DELAY(self.OPEN,1))
                    / IFELSE((temp1 > temp2) & (temp1 > temp3), temp4, 
                             IFELSE((temp2 < temp3) & (temp2 < temp1),temp4,temp3 + temp5))
                    * MAX(temp1, temp2))

    
    
    def alpha_138(self):
        # ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        return ((RANK(DECAYLINEAR(DELTA((((self.LOW * 0.7) + (self.VWAP *0.3))), 3), 20)) 
                 -TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(self.LOW, 8), 
                                                 TSRANK(MEAN(self.VOLUME,60), 17), 5), 19), 16), 7)) * -1)
    
    def alpha_139(self,N=10):
        # (-1 * CORR(OPEN, VOLUME, 10))
        return (-1 * CORR(self.OPEN, self.VOLUME, N))
    
    def alpha_140(self, N=8,M=60):
        # MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), 
        #     TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
        return MIN(RANK(DECAYLINEAR(((RANK(self.OPEN) + RANK(self.LOW)) - (RANK(self.HIGH) + RANK(self.CLOSE))), N)),
                   TSRANK(DECAYLINEAR(CORR(TSRANK(self.CLOSE, N), TSRANK(MEAN(self.VOLUME,M), 20), N), 7), 3))
    
    def alpha_141(self,N=15):
        # (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        return (RANK(CORR(RANK(self.HIGH), RANK(MEAN(self.VOLUME,N)), 9))* -1)
    
    def alpha_142(self, N=5):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        return (((-1 * RANK(TSRANK(self.CLOSE, 2*N))) 
                 * RANK(DELTA(DELTA(self.CLOSE, 1), 1))) 
                * RANK(TSRANK((self.VOLUME/MEAN(self.VOLUME,4*N)), N)))
    
    def alpha_143(self):
        # CLOSE > DELAY(CLOSE, 1)?(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * SELF : SELF
        # 这个因子没有意义
        # 因子构成逻辑是，当价格上涨是，当期因子值等于当期收益率*上期因子值， 
        #                当价格下跌时，当期因子值等于上期因子值
        # 因为收益率是一个小于1的数值，所以因子值始终逼近0。这个因子逻辑没有意义
        # def func(s):
        #     delay = s.shift().fillna(1)
        #     temp1 = s - delay
        #     temp2 = temp1 / delay
        #     res = 1
        #     reslist = []
        #     for i in range(len(s)):
        #         if temp1[i] > 0:
        #             res = temp2[i] * res
        #         reslist.append(res)
        #     return reslist
        
        # alpha = CLOSE.apply(func)                
        return 0
    
    def alpha_144(self,N=20):
        # SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        return (SUMIF(ABS(self.CLOSE/DELAY(self.CLOSE,1)-1)/self.AMOUNT,N,self.CLOSE<DELAY(self.CLOSE,1))
                /COUNT(self.CLOSE<DELAY(self.CLOSE,1),N))
    
    
    def alpha_145(self, N=9,M=26,K=12):
        # (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        return (MEAN(self.VOLUME,N)-MEAN(self.VOLUME,M))/MEAN(self.VOLUME,K)*100
    
    
    def alpha_146(self,N=61,M=20):
        # (MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)
        #  *((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))
        #  /SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)
        #        -((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60) )     
        # 原因子逻辑有问题，少了第三行第一个sma的参数，而且逻辑中有明显漏洞，例如:ret - (ret -sma(ret,61,2))
        # 这里随意改写一下。以后再看
        temp2 = SMA(self.RET, N,2)
        temp1 = self.RET - temp2
        return MEAN(temp1, M) * temp1 / (temp2**2)
    
    
    def alpha_147(self,N=12):
        # REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
        return REGBETA(MEAN(self.CLOSE,N),SEQUENCE(N))
    
    
    def alpha_148(self,N=60,M=9,K=14):
        # ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        return ((RANK(CORR((self.OPEN), SUM(MEAN(self.VOLUME,N), M), 6)) < RANK((self.OPEN - TSMIN(self.OPEN, K)))) * -1)
    
    
    def alpha_149(self):
        # REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,
        #                BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),
        #         FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,
        #                BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
        return 0
    
    
    def alpha_150(self):
        # (CLOSE + HIGH + LOW)/3 * VOLUME
        return (self.CLOSE + self.HIGH + self.LOW) / 3 * self.VOLUME
    
    
    def alpha_151(self, N=20):
        # SMA(CLOSE-DELAY(CLOSE,20),20,1)
        return SMA(self.CLOSE-DELAY(self.CLOSE,N) ,N, 1)
    
       
    def alpha_152(self,N=9,M=12,K=26):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-
        #     MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
        return SMA(MEAN(DELAY(SMA(DELAY(self.CLOSE/DELAY(self.CLOSE,N),1),N,1),1),M)
                   -MEAN(DELAY(SMA(DELAY(self.CLOSE/DELAY(self.CLOSE,N),1),N,1),1),K),N,1)
    
    
    def alpha_153(self,N=3):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 
        return (MEAN(self.CLOSE,N)+MEAN(self.CLOSE,2*N)+MEAN(self.CLOSE,3*N)+MEAN(self.CLOSE,4*N))/4
    
    
    def alpha_154(self,N=16,M=18):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) 
        return (((self.VWAP - MIN(self.VWAP, N))) < (CORR(self.VWAP, MEAN(self.VOLUME,180), M)))
    
    
    def alpha_155(self,N=13,M=27,K=10):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
        temp1 = SMA(self.VOLUME,N,2) - SMA(self.VOLUME,M,2)
        return temp1 - SMA(temp1,K,2)
    
    
    def alpha_156(self,N=5,M=3,P=0.15):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1)
        return (MAX(RANK(DECAYLINEAR(DELTA(self.VWAP, N), M)),
                    RANK(DECAYLINEAR(((DELTA(((self.OPEN * P) + (self.LOW *(1-P))),2)
                                       /((self.OPEN * P) + (self.LOW *(1-P)))) * -1), M))) * -1)
    
    
    def alpha_157(self,N=5):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5))
        return (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((self.CLOSE - 1), N))))), 2), 1)))), 1), N) 
                + TSRANK(DELAY((-1 * self.VWAP), 6), N))
    
    
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        temp = SMA(self.CLOSE, 15, 2)
        return ((self.HIGH - temp) - (self.LOW - temp)) / self.CLOSE
    
    
    def alpha_159(self,N=6):
        # (((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24
        #   +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24
        #   +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100
        #  /(6*12+6*24+12*24))
        temp1 = MIN(self.LOW,DELAY(self.CLOSE,1))
        temp2 = MAX(self.HIGH,DELAY(self.CLOSE,1)) - temp1
        return (((self.CLOSE - SUM(temp1,N)) / SUM(temp2, N) * 8 * (N**2)
                + (self.CLOSE - SUM(temp1,2*N)) / SUM(temp2, 2*N) * 4 * (N**2)
                + (self.CLOSE - SUM(temp1,4*N)) / SUM(temp2, 4*N) * 4 * (N**2))*100 / (14 * (N**2)))

    
    def alpha_160(self,N=20):
        # SMA(IFELSE(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)
        return SMA(IFELSE(self.CLOSE<=DELAY(self.CLOSE,1),STD(self.CLOSE,N),0),N,1)
    

    def alpha_161(self, N=12):
        # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
        return MEAN(MAX(MAX((self.HIGH-self.LOW),ABS(DELAY(self.CLOSE,1)-self.HIGH)),ABS(DELAY(self.CLOSE,1)-self.LOW)),N) 
    
    
    def alpha_162(self,N=12):
        # ((SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #  /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        #  -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #       /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
        # /(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #       /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)
        #   -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #        /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)))
        temp1 = self.CLOSE-DELAY(self.CLOSE,1)
        temp4 = SMA(MAX(temp1,0),N,1)
        temp5 = SMA(ABS(temp1),N,1)
        temp6 = temp4 / temp5 *100 
        return (temp6 - MIN(temp6,N)) / (MAX(temp6, N) -  MIN(temp6,N))
    
    
    def alpha_163(self):
        # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
        return RANK(((((-1 * self.VWAP) * MEAN(self.VOLUME,20)) * self.VWAP) * (self.HIGH - self.CLOSE)))
    
    
    def alpha_164(self,N=12,M=13):
        # SMA((IFELSE((CLOSE>DELAY(CLOSE,1)),1/(CLOSE-DELAY(CLOSE,1)),1)
        #      -MIN(IFELSE((CLOSE>DELAY(CLOSE,1)),1/(CLOSE-DELAY(CLOSE,1)),1),12))/(HIGH-LOW)*100,13,2)
        temp1 = self.CLOSE - DELAY(self.CLOSE,1)
        temp2 = IFELSE(temp1 > 0, 1/temp1, 1)
        return SMA((temp2 - MIN(temp2,N)) / (self.HIGH - self.LOW) * 100, M, 2)
   
    
    def alpha_165(self,N=48):
        # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
        # return MAX(SUMAC(self.CLOSE-MEAN(self.CLOSE,N)))-MIN(SUMAC(self.CLOSE-MEAN(self.CLOSE,N)))/STD(self.CLOSE,N)  
        return 0
    
    def alpha_166(self,N=20):
        # 原公式有错误，随便改了改
        # -20*(20-1)**1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)
        # /((20-1)*(20-2)*(SUM(MEAN(CLOSE/DELAY(CLOSE,1),20)**2,20))**1.5)
          
        return (-N*(N-1)**1.5*SUM(self.RET-MEAN(self.RET,N),N)
                /((N-1)*(N-2)*(SUM(MEAN(self.CLOSE/DELAY(self.CLOSE,1),N)**2,N))**1.5))
    
    
    def alpha_167(self,N=12):
        # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
        temp = self.CLOSE-DELAY(self.CLOSE,1)
        return SUM(IFELSE(temp>0,temp,0),N)
    
    
    def alpha_168(self,N=20):
        # (-1*VOLUME/MEAN(VOLUME,20))
        return (-1*self.VOLUME/MEAN(self.VOLUME,N))
    
    
    def alpha_169(self,N=9,M=12,K=26):
        # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
        temp = DELAY(SMA(self.CLOSE-DELAY(self.CLOSE,1),N,1),1)
        return SMA(MEAN(temp,M)-MEAN(temp,K),10,1)
    
    
    def alpha_170(self,N=20,M=5):
        # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /
        # 5))) - RANK((VWAP - DELAY(VWAP, 5))))
        return ((((RANK((1 / self.CLOSE)) * self.VOLUME) / MEAN(self.VOLUME,N))
                 * ((self.HIGH * RANK((self.HIGH - self.CLOSE))) 
                    / (SUM(self.HIGH, M) / M))) 
                - RANK((self.VWAP - DELAY(self.VWAP, M))))
    
 
    def alpha_171(self):
        # ((-1 * ((LOW - CLOSE) * (OPEN**5))) / ((CLOSE - HIGH) * (CLOSE**5)))
        return ((-1 * ((self.LOW - self.CLOSE) * (self.OPEN**5))) / ((self.CLOSE - self.HIGH) * (self.CLOSE**5)))
    
    
    def alpha_172(self,N=14,M=6):
        # MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #          -SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))
        #      /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #        +SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14)) *100,6)
        temp1 = (self.LD>0) & (self.LD>self.HD)
        temp2 = (self.HD>0) & (self.LD>self.HD)
        temp3 = SUM(IFELSE(temp1,self.LD,0),N)*100 / SUM(self.TR,N)
        temp4 = SUM(IFELSE(temp2,self.HD,0),N)*100 / SUM(self.TR,N)
        return MEAN(ABS(temp3 - temp4) / (temp3 + temp4) * 100, M)

    
    
    def alpha_173(self,N=13):
        # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
        temp = SMA(self.CLOSE,N,2)
        return 3*temp - 2*SMA(temp,N,2) + SMA(SMA(SMA(LOG(self.CLOSE),N,2),N,2),N,2)
    
    
    def alpha_174(self,N=20):
        # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        return SMA(IFELSE(self.CLOSE>DELAY(self.CLOSE,1),STD(self.CLOSE,N),0),N,1)
    
    
    def alpha_175(self,N=6):
        # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
        return MEAN(MAX(MAX((self.HIGH-self.LOW),ABS(DELAY(self.CLOSE,1)-self.HIGH)),ABS(DELAY(self.CLOSE,1)-self.LOW)),N)
    
    
    def alpha_176(self,N=12,M=6):
        # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
        return CORR(RANK(((self.CLOSE - TSMIN(self.LOW, N)) / 
                          (TSMAX(self.HIGH, N) - TSMIN(self.LOW,N)))), RANK(self.VOLUME), M)
    
    
    def alpha_177(self,N=20):
        # ((20-HIGHDAY(HIGH,20))/20)*100
        return ((N-HIGHDAY(self.HIGH,N))/N)*100
    
    
    def alpha_178(self):
        # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
        return (self.CLOSE-DELAY(self.CLOSE,1))/DELAY(self.CLOSE,1)*self.VOLUME
    
    
    def alpha_179(self,N=50):
        # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
        return (RANK(CORR(self.VWAP, self.VOLUME, 4)) *RANK(CORR(RANK(self.LOW), RANK(MEAN(self.VOLUME,N)), 12))) 
    
    
    def alpha_180(self,N=20,M=7):
        # ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) 
        return IFELSE((MEAN(self.VOLUME,N) < self.VOLUME), 
                      (-1 * TSRANK(ABS(DELTA(self.CLOSE, M)), 3*N)) * SIGN(DELTA(self.CLOSE, M)), 
                       (-1 * self.VOLUME))
    

    def alpha_181(self):
      
        return 0
    
    
    def count_cond_182(self, x):
        num = 0
        for i in x:
            if i == np.True_:
                num += 1
        return num
    
    def alpha_182(self):
        ##### COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20 #####
        # cond1 = (self.close>self.open_price)
        # cond2 = (self.benchmark_open_price>self.benchmark_close_price)
        # cond3 = (self.close<self.open_price)
        # cond4 = (self.benchmark_open_price<self.benchmark_close_price)
        # func1 = lambda x: np.asarray(x) & np.asarray(cond2)
        # func2 = lambda x: np.asarray(x) & np.asarray(cond4)
        # cond = cond1.apply(func1)|cond3.apply(func2)
        # count = pd.rolling_apply(cond, 20, self.count_cond_182)
        # alpha = (count/20).iloc[-1,:]
        # alpha=alpha.dropna()
        return 0
    
    
    def alpha_183(self):
        # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
        # 公式有问题
        return 0
    
    
    def alpha_184(self):
        # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
        return (RANK(CORR(DELAY((self.OPEN - self.CLOSE), 1), self.CLOSE, 200)) + RANK((self.OPEN - self.CLOSE)))

    
    def alpha_185(self):
        # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
        return RANK((-1 * ((1 - (self.OPEN / self.CLOSE))**2)))
    
    
         
    def alpha_186(self,N=14,M=6):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #           -SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))
        #       /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #         +SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        #  +DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #                  -SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))
        #              /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #                +SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        
        temp1 = (self.LD>0) & (self.LD>self.HD)
        temp2 = (self.HD>0) & (self.LD>self.HD)
        temp3 = SUM(IFELSE(temp1,self.LD,0),N)*100 / SUM(self.TR,N)
        temp4 = SUM(IFELSE(temp2,self.HD,0),N)*100 / SUM(self.TR,N)
        temp5 = MEAN(ABS(temp3 - temp4) / (temp3 + temp4) * 100, M)
        return (temp5 + DELAY(temp5,M)) / 2
    
    
    def alpha_187(self,N=20):
        # SUM(IFELSE(OPEN<=DELAY(OPEN,1),0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
        return SUM(IFELSE(self.OPEN <= DELAY(self.OPEN,1),0,
                          MAX((self.HIGH-self.OPEN),(self.OPEN-DELAY(self.OPEN,1)))),N)

    
    def alpha_188(self,N=11):
        # ((HIGH - LOW - SMA(HIGH-LOW, 11, 2)) / SMA(HIGH-LOW, 11, 2))*100      
        temp = self.HIGH - self.LOW
        return (temp - SMA(temp,N,2))  / SMA(temp,N,2) *100
    
    
    def alpha_189(self,N=6):
        # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
        return MEAN(ABS(self.CLOSE-MEAN(self.CLOSE,N)),N)
    
    
    def alpha_190(self):
        # LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))**(1/20)-1),20)-1)
        #     *(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,
        #             CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1))
        #     /((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1),20))
        #       *(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,
        #               CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))**(1/20)-1))))      
        
        temp1 = self.CLOSE/DELAY(self.CLOSE)-1
        temp2 = (self.CLOSE/DELAY(self.CLOSE,19))**(1/20)
        return LOG(((COUNT(temp1 > (temp2-1), 20) - 1) * SUMIF((temp1 - temp2-1)**2, 20, temp1 < (temp2-1)))
                   /((COUNT(temp1 < (temp2-1), 20)) * SUMIF((temp1 - (temp2 - 1))**2, 20, temp1 > (temp2-1))))

    def alpha_191(self, n=20, m=5):
        # (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####        
        return (CORR(MEAN(self.VOLUME,n), self.LOW, m) + ((self.HIGH + self.LOW) / 2)) - self.CLOSE
    

    def alpha_192(self, N=26):
        # 意愿指标	BR=N日内（当日最高价－昨日收盘价）之和 / N日内（昨日收盘价－当日最低价）之和×100 n设定为26
        return SUM(self.HIGH - self.CLOSE, N) / SUM(self.CLOSE - self.LOW, N) * 100

    def alpha_193(self, N=20, M=120):
        # ARBR 因子 AR 与因子 BR 的差
        return (SUM(self.HIGH - self.OPEN, N) / SUM(self.OPEN - self.LOW, N)*100 
                - SUM(self.HIGH - self.CLOSE, N) / SUM(self.CLOSE - self.LOW, N) * 100)

    def alpha_194(self, N=20):
        # 20日收益方差
        return VAR(self.RET, N)
    
    
    def alpha_195(self, N=20):
        # 收益的20日偏度
        return SKEW(self.RET, N)


    def alpha_196(self, N=20):
        # 收益的20日峰度
        return KURT(self.RET, N)


    def alpha_197(self, N=60):
        # 60日收益方差
        return VAR(self.RET, N)
    
    
    def alpha_198(self, N=60):
        # 收益的60日偏度
        return SKEW(self.RET, N)


    def alpha_199(self, N=60):
        # 收益的60日峰度
        return KURT(self.RET, N)


    def alpha_200(self, N=20):
        # 20日夏普比率
        return SHARPE(self.RET, N)

    def alpha_201(self, N=60):
        # 60日夏普比率
        return SHARPE(self.RET, N)

    def alpha_202(self, N=120):
        # 120日夏普比率
        return SHARPE(self.RET, N)

    def alpha_203(self, N=120):
        # 120日收益方差
        return VAR(self.RET, N)
    
    
    def alpha_204(self, N=120):
        # 收益的120日偏度
        return SKEW(self.RET, N)


    def alpha_205(self, N=120):
        # 收益的120日峰度
        return KURT(self.RET, N)

        
    def alpha_206(self,N=10):
        # CLOSE-DELAY(CLOSE,5)
        return self.CLOSE / DELAY(self.CLOSE,N) - 1
    
    def alpha_207(self,N=20):
        # CLOSE-DELAY(CLOSE,5)
        return self.CLOSE / DELAY(self.CLOSE,N) - 1
    
    def alpha_208(self,N=10):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return SUM(IFELSE(self.RET>0,self.RET, 0), N) / SUM(self.RET, N)

    def alpha_209(self,N=30):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return SUM(IFELSE(self.RET>0,self.RET, 0), N) / SUM(self.RET, N)
        
    def alpha_210(self,N=100):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return SUM(IFELSE(self.RET>0,self.RET, 0), N) / SUM(self.RET, N)
   
    def alpha_211(self, N=10):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)
    
    def alpha_212(self, N=30):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)
    
    def alpha_213(self, N=100):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)

    def alpha_214(self, N=10):
        # 日内动量因子
        return MEAN(self.CLOSE / self.OPEN - 1, N)
    
    def alpha_215(self, N=30):
        # 日内动量因子
        return MEAN(self.CLOSE / self.OPEN - 1, N)

    def alpha_216(self, N=100):
        # 日内动量因子
        return MEAN(self.CLOSE / self.OPEN - 1, N)

    def alpha_217(self, N=10):
        # k线均线因子
        return self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_218(self, N=30):
        return self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_219(self, N=100):
        return self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_220(self, N=10):
        # 快慢均线趋势因子
        return MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1
    
    def alpha_221(self, N=30):
        return MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1
    
    def alpha_222(self, N=50):
        return MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1

    def alpha_223(self, N=10):
        # 日内累计振幅因子
        # MEAN((2 * (HIGH - LOW) * SIGN(CLOSE - OPEN) - (CLOSE - OPEN)) / CLOSE, N)
        return MEAN((2 * (self.HIGH - self.LOW) * SIGN(self.CLOSE - self.OPEN) 
                     - (self.CLOSE - self.OPEN)) / self.CLOSE, N)

    def alpha_224(self, N=30):
        # 日内累计振幅因子
        return MEAN((2 * (self.HIGH - self.LOW) * SIGN(self.CLOSE - self.OPEN) 
                     - (self.CLOSE - self.OPEN)) / self.CLOSE, N)

    def alpha_225(self, N=10):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return  MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)


    def alpha_226(self, N=40):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return  MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

    def alpha_227(self, N=100):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return  MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

        
    def alpha_228(self, N=10):
        # 稳健动量因子
        temp = RANK(self.RET,pct=False)
        n = temp.shape[1]      
        return MEAN( (temp - (n +1)/2) / (((n+1)*(n-1)/12)**0.5), N)

    def alpha_229(self, N=40):
        # 稳健动量因子
        temp = RANK(self.RET,pct=False)
        n = temp.shape[1]      
        return MEAN( (temp - (n +1)/2) / (((n+1)*(n-1)/12)**0.5), N)


    def alpha_230(self, N=5):
        # GK 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        return (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5

    def alpha_231(self, N=30):
        # RS 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        return (252 * MEAN( h * (h - c) - l * (l - c), N)) ** 0.5

    def alpha_232(self, N=30):
        # PK 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        return (252 / 4 / np.long(2) * MEAN( (h - l) ** 2, N)) ** 0.5


    def alpha_233(self, N=30):
        # cci
        typicalPrice = (self.HIGH + self.LOW + self.CLOSE) /3
        avgTypicalPrice = MEAN(typicalPrice, N)
        avgDev = MEAN(abs(typicalPrice - avgTypicalPrice), N)
        return (typicalPrice - avgTypicalPrice) /  avgDev / 0.015
    
    def alpha_234(self, N=14):
        # mfi
        typicalPrice = (self.HIGH + self.LOW + self.CLOSE) /3
        diff = typicalPrice.diff()
        positiveMF = typicalPrice * self.VOLUME * (diff > 0)
        negativeMF = typicalPrice * self.VOLUME * (diff < 0)
        MFratio = MEAN(positiveMF,N) / MEAN(negativeMF,N)
        # MFratio[MFratio == inf] = 100
        # mfi = 100 - 100 / (1 + MFratio)
        return 100 - 100 / (1 + MFratio)
    
    def alpha_235(self,N=9):
        # kd_ K
        L = TSMIN(self.LOW,N)
        RSV = (self.CLOSE - L) / (TSMAX(self.HIGH,N) - L)
        return RSV.ewm(com=2).mean()
    
    def alpha_236(self,N=9):
        # kd _D
        L = TSMIN(self.LOW,N)
        RSV = (self.CLOSE - L) / (TSMAX(self.HIGH,N) - L)
        return (RSV.ewm(com=2).mean()).ewm(com=2).mean()
    
    def alpha_237(self,N=9):
        # kd _J
        L = TSMIN(self.LOW,N)
        RSV = (self.CLOSE - L) / (TSMAX(self.HIGH,N) - L)
        K = RSV.ewm(com=2).mean()
        D = (RSV.ewm(com=2).mean()).ewm(com=2).mean()
        return 3 * K - 2 * D
    
    def alpha_238(self, N=5):
        factor = self.CLOSE / DELAY(self.CLOSE,N) - self.CLOSE/DELAY(self.CLOSE,int(2*N))
        return factor * -1

    def alpha_239(self, N=10):
        factor = self.CLOSE / DELAY(self.CLOSE,N) - self.CLOSE/DELAY(self.CLOSE,int(2*N))
        return factor * -1
    
    def alpha_240(self, N=20):
        factor = self.CLOSE / DELAY(self.CLOSE,N) - self.CLOSE/DELAY(self.CLOSE,int(2*N))
        return factor
  
    def alpha_241(self, N=30):
        # boll_index, hp=5
        std = STD(self.CLOSE, N)
        return (self.CLOSE - ( MEAN(self.CLOSE, N) - std ) ) / (2 * std)

    def alpha_242(self, N=15):
        # boll_index, hp=5
        std = STD(self.CLOSE, N)
        return (self.CLOSE - ( MEAN(self.CLOSE, N) -  std ) ) / (2 * std)
    
    def alpha_243(self, N=60):
        # 均线发散1
        ema = self.CLOSE.ewm(adjust=False, alpha=2/(N + 1), ignore_na=True).mean()
        ma = MEAN(self.CLOSE, N)
        return (ema * 3 - ma * 2) / ma
        
    def alpha_244(self, N=30):
        # 均线发散2
        ema = self.CLOSE.ewm(adjust=False, alpha=2/(N + 1), ignore_na=True).mean()
        ma = MEAN(self.CLOSE, N)
        return (ema * 3 - ma * 2) / ma
    
    def alpha_245(self, N=10):
        # 均线发散2
        ema = self.CLOSE.ewm(adjust=False, alpha=2/(N + 1), ignore_na=True).mean()
        ma = MEAN(self.CLOSE, N)
        return (ema * 3 - ma * 2) / ma
    
        



