# -*- coding: utf-8 -*-
"""
因子组合基础函数

Created on Wed Nov  8 09:41:04 2023

Edited on Wed Nov  8 09:41:04 2023


@author: oOoOo_Andra
"""
import pandas as pd
import numpy as np
import scipy.optimize as sco
import bottleneck as bn
#%% 过滤
def factor_filter(corr_df, threshold = 0.6):
    f_filter = []
    for col,row in [(s, t) for s in range(len(corr_df)) for t in range(s+1,len(corr_df))]:
        if (col in f_filter) or (row in f_filter):
            continue
        if abs(corr_df[corr_df.index[col]][corr_df.index[row]]) > threshold:
            f_filter += [row]
    return [corr_df.index[s] for s in range(len(corr_df)) if not s in f_filter]


def factor_filter_sr(corr_df, dfret, N=5, threshold=0.5):
    '''
    通过sr公式做过滤
    '''
    ret_ptf = pd.DataFrame(dfret[corr_df.index[0]])
    
    f_filter = [corr_df.index[0]]
    
    for i in range(1,len(corr_df)):
        # if corr_df.iloc[i,i-1] > 
        if len(f_filter) < N:
            ret_new = dfret[corr_df.index[i]]
             
            ret_ptf_new = pd.concat([ret_ptf.mean(axis=1), ret_new],axis=1)
            
            corr = ret_ptf_new.corr().iloc[1,0]
            
            sr_ptf = (ret_ptf_new.mean() / ret_ptf_new.std()).tolist()
            
            if (corr < threshold) & (sr_ptf[0] * corr < sr_ptf[1]) :
                
                ret_ptf = pd.concat([ret_ptf, ret_new],axis=1)
                
                f_filter.append(corr_df.index[i])
            else:
                ret_ptf_new = pd.DataFrame()
        else:
            print('finished')
            break
    return f_filter

#%% 因子组合
def HalfDecay(H:int, T:int):
    '''
    半衰权重

    Parameters
    ----------
    H : int
        半衰期,每经过 H 期（向过去前推 H 期），权重变为原来的一半
    T : int
        过去的期数

    Returns
    -------
    res : TYPE
        半衰权重

    '''
    t_range = np.arange(1, T + 1)
    #半衰权重
    Wt = 2 ** ( (t_range - T - 1) / H)
    # 归一化
    res = Wt / Wt.sum()
    
    return res


def objective_functions(method='min_sr'):
    if method == 'min_sr':
        pass
        
def opt_stats(weights,daily_profit):
    '''
    weights:np.array
    daily_profit: dataFrame
    参考文献：https://zhuanlan.zhihu.com/p/60499205
    '''

    weights = np.array(weights)
    pret = np.sum(daily_profit.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(daily_profit.cov() * 252, weights)))
    return pret,pvol,pret/pvol

def max_sp(weights, daily_profit):
    weights = np.array(weights)
    
    portfolio_return = np.sum(daily_profit.mean() * weights) * 252
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_profit.cov() * 252, weights)))

    return -portfolio_return/portfolio_volatility


def max_sortino(weights,daily_profit):
    '''
    weights: np.array
    daily_profit: DataFrame
    '''
    weights = np.array(weights)
    
    portfolio_return = np.sum(daily_profit.mean() * weights) * 252
    
    # r_p_min = np.sum(daily_profit[daily_profit < 0].mean() * weights) * 252
    
    # 下行标准差
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.minimum(daily_profit, 0).cov() * 252, weights)))
    
    return -portfolio_return / portfolio_volatility

def min_volatility(weights,daily_profit):
    weights = np.array(weights)
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_profit.cov() * 252, weights)))
    
    return portfolio_volatility

def max_upside_volatility(weights,daily_profit):
    weights = np.array(weights)
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.maximum(daily_profit, 0).cov() * 252, weights)))
    
    return -portfolio_volatility

def min_downside_volatility(weights,daily_profit):
    weights = np.array(weights)
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.minimum(daily_profit, 0).cov() * 252, weights)))
    
    return portfolio_volatility



#%% 收益计算
def calc_return(dfweight, ret, jumpRet, dfcost, reweight=True):
    '''根据生成的权重计算相应的收益'''
    dfweight1 = dfweight.fillna(method='ffill')
    if reweight:
    
        dfweight2 = dfweight1.div(dfweight1.abs().sum(axis=1), axis=0)
    else:
        dfweight2 = dfweight1
    
    dfsig = dfweight2.diff()
    
    dfret = (dfweight2.shift() * ret.fillna(0) -\
        jumpRet * (dfsig.shift().fillna(0) * 1) - \
        dfsig.shift().abs().fillna(0) * dfcost)
    
    cumret = dfret.sum(axis=1).cumsum()
    return dfret, cumret, dfweight2
        


   
def calc_return2(dfweight, ret, jumpRet, dfcost, th=0.1):
    '''
    根据生成的权重计算相应的收益
    最大单品种上限10%
    '''
    dfweight1 = dfweight.fillna(method='ffill')
    dfweight2 = dfweight1.div(dfweight1.abs().sum(axis=1), axis=0)
    
    dfweight2 = np.minimum(dfweight2.abs(),th) * np.sign(dfweight2)
    
    dfsig = dfweight2.diff()
    
    dfret = (dfweight2.shift() * ret.fillna(0) -\
        jumpRet * (dfsig.shift().fillna(0) * 1) - \
        dfsig.shift().abs().fillna(0) * dfcost)
    
    cumret = dfret.sum(axis=1).cumsum()
    return dfret, cumret


        
#%% 权重输出
def signal_generator(weight: pd.Series, factors_signal_all: pd.DataFrame):
    '''

    Parameters
    ----------
    weight : pd.Series
            factor    hp   factorName  test
    alpha_206_param3  20D  alpha_206   433  |   0.114538
    alpha_208_param4  20D  alpha_208   440  |   0.061183
    alpha_211_param0  3D   alpha_211   98   |   0.020606
    
    
    factors_signal_all : pd.DataFrame
        DESCRIPTION.
                                 A  AG  AL  AP  AU  B  ...  SS  TA  UR  V  Y  ZN
    date       factor                                   ...                      
    2016-01-04 alpha_206_param0   0   0   0   0   0  0  ...   0   0   0  0  0   0
    2016-01-05 alpha_206_param0   0   0   0   0   0  0  ...   0   0   0  0  0   0
    2016-01-06 alpha_206_param0   0   0   0   0   0  0  ...   0   0   0  0  0   0
    2016-01-07 alpha_206_param0   0   0   0   0   0  0  ...   0   0   0  0  0   0
    2016-01-08 alpha_206_param0   0   0   0   0   0  0  ...   0   0   0  0  0   0
    
    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    '''
    signal = pd.DataFrame(0, index=factors_signal_all.index.unique(level=0), 
                          columns=factors_signal_all.columns).sort_index()
    
    try:
        weight = weight.to_frame('weight')
    except:
        pass
    
    weight['hp'] = weight.index.get_level_values('hp').str.strip('D').astype('int').tolist()
    
    try:
        weight.reset_index(level=0, inplace=True)
    except:
        pass

    for i in range(len(weight)):
        factorName,w, hp,= weight.iloc[i,:]
        signal_f = factors_signal_all.xs(factorName, level='factor')
        signal_p = signal_f.rolling(hp).sum() / hp
        signal += (signal_p.div(signal_f.abs().sum(axis=1), axis=0) * w).fillna(0)
        
    return signal




def weightReconstrution(dfpos_longshort1,dfgroupII,groupWeightTh=0.2, mode='spread'):
    dfposG = pd.DataFrame()
    for i in range(len(dfgroupII)):
        name = dfgroupII.iloc[i,0]
        member = dfgroupII.iloc[i,3:].dropna().tolist()  
        dfposG[name] = dfpos_longshort1[member].sum(axis=1)
        
    # over = (dfposG < -groupWeightTh)     
    # overWeight = (over * dfposG).sum(axis=1) - over.sum(axis=1) * groupWeightTh
    
    # if direction == 'long':
    #     dfposGedited = np.minimum(dfposG, groupWeightTh)
    # elif direction == 'short':
    #     dfposGedited = np.maximum(dfposG, -groupWeightTh)
    
    # elif direction == 'longshort':
    #     dfposGedited = np.minimum(dfposG.abs(), groupWeightTh) * np.where(dfposG>0, 1, -1)
    dfposGedited = np.minimum(dfposG.abs(), groupWeightTh) * np.where(dfposG>0, 1, -1)
    
    if mode == 'spread':
        over = (dfposG.abs() < groupWeightTh)   
        spread = (1 - dfposGedited.abs().sum(axis=1) ) / (over).sum(axis=1)
        '''这里sprad权重之后，可能会出现原本权重在20一下的组变成20以上，后续懒得处理了，应该是大差不差的'''
        dfposGedited = dfposGedited + (over.apply(lambda x: x * spread)) * np.where(dfposG>0, 1, -1)
    
    dfposG2 = pd.DataFrame()
    for i in range(len(dfgroupII)):
        name = dfgroupII.iloc[i,0]
        member = dfgroupII.iloc[i,3:].dropna().tolist()
        weight = dfposGedited[name]
        dfposG2[member] = (dfpos_longshort1[member].apply(lambda x:(x / dfpos_longshort1[member].sum(axis=1)))).apply(lambda x: x* weight)
            
    return dfposG2

def output_position(dfweightMsr, capital, future_info, prices, fileName,
                    filepath_templete, filepath_tradePosition):
    cols = dfweightMsr.columns
    
    point = future_info.loc[cols, 'point']
    
    min_lots = future_info.loc[cols, 'minTradeLots']
    
    dfcap = prices[cols] * point * min_lots
    
    dflots = (capital * dfweightMsr / dfcap).fillna(0).round() * min_lots
    
    position = pd.read_csv(filepath_templete)
    
    position_columns = position.columns
    
    position.set_index('Exchange Symbol', drop=True, inplace=True)
    
    position.loc[cols, 'Target Position'] = dflots.iloc[-1,:].astype(int)
    
    position.loc[cols, 'Current Position'] = dflots.iloc[-2,:].astype(int)
    
    pos_trade = dflots.iloc[-1,:] - dflots.iloc[-2,:]
    
    position.loc[cols, 'Trade Quantity'] = pos_trade.abs().astype(int)
    
    position.reset_index(inplace=True)
    
    position = position[position_columns]
    
    position.to_csv(f'{filepath_tradePosition}{fileName}',index=False)
    
    
    outputCap = dflots * prices[cols] * point

    print(f"""position files generated @ {filepath_tradePosition}{fileName}""")
    
    print(f"""total position = {round(outputCap.abs().sum(axis=1)[-1]/1000000,2)}unit""")
    print(f"""net exposure = {round(outputCap.sum(axis=1)[-1]/1000000,2)}unit""")
    
    print('#' * 40)

    
    return outputCap
    