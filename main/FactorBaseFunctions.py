# -*- coding: utf-8 -*-
"""
多因子模型基础函数

Created on Wed Nov  8 09:18:39 2023

Edited on Wed Nov  8 09:18:39 2023


@author: oOoOo_Andra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

#%% 标准化函数：
def standardize_ts(s,ty=2):
    '''
    请使用filter_extreme_normalize
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    data=s.dropna().copy()
    if int(ty)==1:
        re = (data - data.min())/(data.max() - data.min())
    elif ty==2:
        re = (data - data.mean())/data.std()
    elif ty==3:
        re = data/10**np.ceil(np.log10(data.abs().max()))
    return re

def standardize_sec(factor, ty=2):
    
    '''
    请使用filter_extreme_normalize
    
    
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    data = factor#.dropna().copy()
    if int(ty)==1:
        re = data.sub(data.min(axis=1), axis=0).div(data.max(axis=1) - data.min(axis=1), axis=0)
        
    elif ty==2:
        re = data.sub( data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        
    elif ty==3:
        re = data.div(10**np.ceil(np.log10(data.abs().max(axis=1))), axis=0)
    return re
    


def filter_extreme_normalize(factor, axis=1, method=2):
    '''
    正态化
    Parameters
    ----------

    factor: pd.DataFrame
            为因子数据， 格式为：
            code               A        AG        AL  ...         V         Y        ZN
            tradingday                                ...                              
            2023-07-07  0.009111  0.021889  0.044400  ...  0.061001  0.018669  0.096778
            2023-07-10  0.013092 -0.003132  0.036692  ...  0.046918  0.002484  0.099574
            2023-07-11 -0.000428 -0.003007  0.021454  ...  0.041642  0.011998  0.086422
            2023-07-12 -0.000900  0.018170  0.020047  ...  0.026405 -0.001653  0.088093
            2023-07-13  0.003394 -0.009821  0.012362  ...  0.006803  0.004047  0.066999
    
    axis: {0 , 1 }, default 1
        Whether to compare by the index (0 or ‘index’) or columns. (1 or ‘columns’). 
        For Series input, axis to match Series index on.
        axis=1, 横截面标准化
        axis=0, 时序标准化
    
    method: int, default 2
        为标准化类型:1 MinMax,2 Standard,3 maxabs 

    Returns
    -------
    same type as caller
    
    '''
    if axis == 'columns':
        axis = 1
    else:
        axis = 0
        
    axis_ = 1 - axis
    
    
    if method == 2:
        mean = factor.mean(axis = axis)
    
        std = factor.std(axis = axis)

        return (factor.sub(mean, axis=axis_)).div(std, axis=axis_)
    
    elif method == 1:
        
        mean = factor.mean(axis = axis)
        
        min_ = factor.min(axis = axis)
        
        max_ = factor.max(axis = axis)
        
        return (factor.sub(mean, axis=axis_)).div(max_ - min_, axis=axis_) 
    
    elif method == 3:
        return factor.div( 10 ** np.ceil(np.log10(factor.abs().max(axis=axis))), axis=axis_)
        
    
    
def filter_extreme_3sigma_series(series, n=3):
    '''
    3个标准差去极值

    Parameters
    ----------
    series : pd.Series
        DESCRIPTION.
    n : float, optional
        标准差倍数. T
        he default is 3.

    Returns
    -------
    res : pd.Series
        DESCRIPTION.

    '''
    mean = series.mean()
    std = series.std()
    up = mean + n * std
    dn = mean - n * std
    res = np.clip(series, dn, up)
    return res

def filter_extreme_3sigma_dataframe(df, k=3,):
    '''
    MAD3
    3倍std去极值
    '''
    res = df.apply(filter_extreme_3sigma_series, args=(k,),axis=1)
    return res

def filter_extreme_MAD_series(series, n=3):
    '绝对值差中位数法'
    med = series.median()
    new_med = ((series - med).abs()).median()
    up = med + n * new_med
    dn = med - n * new_med
    return np.clip(series, up, dn)
    
def filter_extreme_MAD_dataframe(df, k=3):
    res = df.apply(filter_extreme_MAD_series, args=(k,),axis=1)
    return res
        

def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

#%% group方法
def factor_fast_groupon(factors, groupNum=5):
    '很快，但是有瑕疵，bins是线性分割的，会导致各个分组后的个数不一致'
    df_rank = factors.rank(axis=1, ascending=False)
    
    df_rank_1 = df_rank.div(df_rank.max(axis=1), axis=0)
    
    # create a boolean mask to identify NaN values
    nan_mask = np.isnan(factors)
    
    # use np.digitize with zeroed values
    factors_group = np.digitize(df_rank_1, bins=np.linspace(0, 1, groupNum+1), right=True)
    
    # set NaN values back to NaN
    factors_group = np.where(nan_mask, np.nan, factors_group)
   
    return pd.DataFrame(factors_group, columns=factors.columns, index=factors.index)


def grouper(factor_rank, groupNum=5, n=1, method='ceil'):
    """
    给所要的分组打上记号

    Parameters
    ----------
    factor_group_rank : TYPE
        因子rank后的df
    groupNum : TYPE, optional
        分组的数目
    n : TYPE, optional
        需要得到的分组序号 
        The default is 1.
    method : TYPE, optional： 'ceil', 'floor'
        分组数目是上下取整还是向下取整
        如果只做一个多空，可以用ceil
        如果同时做2个多空组合，可以用floor
        The default is 'ceil'.

    Returns
    -------
    factor_group

    """
    if method == 'ceil':
        symbolNum = np.ceil((factor_rank.max(axis=1)) / groupNum)
    else:
        symbolNum = ( (factor_rank.max(axis=1)) // groupNum )
    
    if n < groupNum / 2:    
        factor_group = (((factor_rank.sub(symbolNum * n , axis=0)) <= 0) \
                         & (((factor_rank.sub(symbolNum*(n-1), axis=0)) > 0)))
    elif n > groupNum / 2: 
        temp = factor_rank.rank(axis=1, ascending=False)
        factor_group = (((temp.sub(symbolNum * (groupNum - n + 1) , axis=0)) <= 0) \
                         & (((temp.sub(symbolNum * (groupNum - n), axis=0)) > 0)))
    elif n == groupNum / 2:
        temp = factor_rank.sub(factor_rank.mean(axis=1), axis=0)
        factor_group = temp.abs().sub(symbolNum / 2, axis=0) < 0
        
    return factor_group



#%% 绩效
def performance_ratio(dailyReturn):
    cum_rate = dailyReturn.cumsum()
    try:
        sharpe_ratio = math.sqrt(252) * dailyReturn.mean() / dailyReturn.std()
    except:
        sharpe_ratio = np.nan
    annual_return = 252 * dailyReturn.mean()
    high_point = cum_rate.cummax()
    drowdown = high_point-cum_rate
    max_drowdown = round(drowdown.max(), 3)
    try:
        mar = annual_return /max_drowdown
    except :
        mar = np.nan
    
    res = [round(annual_return, 4), round(sharpe_ratio,4),
           round(max_drowdown,4),   round(mar,4)]
    return res    


def fig_label(dailyReturn, string='return'):
    res = performance_ratio(dailyReturn)
        
    label = f'''{string}
    ar={round(res[0]*100,2)}%, sr={round(res[1],2)},
    mdd={round(res[2]*100,2)}%, mar={round(res[3],2)}'''

    return label,res

def factor_turnover(pos):
    #pos = dff / i
    # pos.fillna(method='ffill',inplace=True)
    #!!! 20340924 把ffill改为0
    pos.fillna(0,inplace=True)
    turnover = pos.diff().abs().sum(axis=1) / (pos.abs().sum(axis=1))
    turnover_avg = turnover.mean(skipna=True)
    return turnover_avg



def factor_corr_plot(df_ret_ptf,fig=True):
    # 因子组合的相关性矩阵作图；
    # Compute the correlation matrix
    corr = df_ret_ptf.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    if fig:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
    
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})    
  
    return corr


def calc_icir_single(factor, ret):
    '''
    [a,b,c,d,e,f] = calc_icir_single(factor, ret)
    '''    
    if factor.shape != ret.shape:
        raise KeyError('Shape is not mapping!')
    else:
        # ic序列
        IC = factor.corrwith(ret, axis=1)
        
        # rankIC序列
        rankIC = factor.rank(axis=1).corrwith(ret.rank(axis=1), axis=1)

        # ic均值
        IC_mean = IC.mean()
        rankIC_mean = rankIC.mean()
        
        # ic标准差
        IC_std = IC.std()
        rankIC_std = rankIC.std()
        
        
        # icir
        ICIR = IC_mean / IC_std * np.sqrt(len(IC))
        rankICIR = rankIC_mean / rankIC_std * np.sqrt(len(IC))
        
        return [IC, IC_mean, ICIR, 
                rankIC, rankIC_mean, rankICIR]


def calc_normal_ic_sec_allFactors(factor, ret, method='normal'):   
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

def calc_icir_multiHP(ic,n=250):
    icir = ic.groupby('cycle').rolling(n).apply(lambda x: x.mean() / x.std())
    return icir    

def show(dataframe,label='',legend=False):
    _, ax = plt.subplots()
    ax.plot(dataframe, label=label)
    if legend:
        ax.legend(loc='best')
    ax.grid(True)
    


#%% 因子测试
def cal_cost(data,future_list,slip=-1):
    '''计算手续费,默认为slip=-1,自己重新设置手续费，需要手动指定
    '''
    product = data.name
    bp = future_list.loc[product,'tick_size'] 

    if slip==-1:
        slip = future_list.loc[product,'slip']
    
    # 判断手续费类型和值
    if future_list.loc[product,'type'] == 1:    
        fee = future_list.loc[product,'commission'] + slip * bp / data
    else:
        fee = (future_list.loc[product,'commission'])/(data * future_list.loc[product,'point'] ) +  slip * bp/ data  
    return fee  


def cal_cost2(data, future_list,capital, slip=-1):
    '''计算手续费,默认为slip=-1,自己重新设置手续费，需要手动指定
    以金额计
    '''
    product = data.name
    bp = future_list.loc[product,'tick_size'] 
    if slip==-1:
        slip = future_list.loc[product,'slip']
    # 判断手续费类型和值
    if future_list.loc[product,'type'] == 1:    
        fee = (future_list.loc[product,'commission'] * data + slip * bp) * future_list.loc[product,'point']
    else:
        fee = pd.Series(future_list.loc[product,'commission'] +  slip * bp * future_list.loc[product,'point'],
                        index=data.index)
    # totalLots = capital / (data.close * future_list.loc[product,'point'] )
    
    return fee


def factor_test_group(factor, rets, cost, groupNum=5, h=5,  factorName='factor', fig=True, fig_path='C:/desktop/'):
    '''
    23Oct16
        增加成本的计算
        增加费后收益的绩效
        更改了图片的输出，全都放入一张fig
    23Oct24
        增加12 45分组
        信号输出
        删除 收益和因子表头对齐 操作
        删除 因子值 阈值 过滤
    23Oct26
        增加所有ls分组费后
        输出的收益序列由累计收益改为每日收益
    '''
    #--- 1 因子名定义
    if hasattr(factor, 'name'):
        factorName = factor.name
    else:
        factorName = factorName

    #--- 2 factor group rank
    factor_group_rank = factor.rank(axis=1, ascending=False).apply(lambda x
                         : pd.cut(x,bins=groupNum, labels=False,duplicates='drop'),axis=1) + 1
    
        
    #--- 3 重构收益率表的表头,计算为了n期的平均收益率
    ret = rets.rolling(h).mean().shift(-h)
    
    #--- 4 loop按照组别循环
    factor_return = pd.DataFrame()
    dict_sig = {}
    
    for i in range(1,groupNum+1):
        # 分组信号
        sig = (factor_group_rank[factor_group_rank == i]) / i
        # 品种每个点的权重
        sig = sig.div(sig.sum(axis=1), axis=0)
        # 信号h分之后，根据持有h期，当天的累积信号
        signal = sig.fillna(0).rolling(h).sum() / h
        # 储存在字典中
        dict_sig[f'group_{i}'] = signal
        # 分组收益
        # factor_return_group = ((ret * sig).sum(axis=1))
        factor_return_group = (signal.shift() * rets).sum(axis=1)
    
        factor_return = pd.concat([factor_return,factor_return_group],axis=1)       

    #--- 5 分组收益
    factor_return.columns = [('group_' + str(i)) for i in range(1, groupNum+1)]

    #--- 6分组多空收益
    factor_return_ls = pd.DataFrame()
   
    factor_return_ls['group_LS_1']  = 0.5 * (factor_return.iloc[:,0] - factor_return.iloc[:,-1]) 
   
    factor_return_ls['group_LS_2']  = 0.5 * (factor_return.iloc[:,1] - factor_return.iloc[:,-2]) 
    
    factor_return_ls['group_LS_12'] = 0.5 * (factor_return.iloc[:,:int(groupNum / 2)].mean(axis=1) 
                                             - factor_return.iloc[:,-int(groupNum / 2):].mean(axis=1)) 
    # 合并
    factor_return = pd.concat([factor_return,factor_return_ls], axis=1)   
    
    #--- 7 多空信号
    # 1-5分组多空信号
    dict_sig['group_LS_1']  = (dict_sig['group_1'].fillna(0) 
                              - dict_sig[f'group_{groupNum}'].fillna(0))
    # 2-4分组多空信号
    dict_sig['group_LS_2']  = (dict_sig['group_2'].fillna(0) 
                              - dict_sig[f'group_{groupNum-1}'].fillna(0))
    # 12 - 45分组多空信号
    dict_sig['group_LS_12'] = dict_sig['group_LS_1'] + dict_sig['group_LS_2']

    list_col = ['LS_1','LS_2','LS_12']
    
    #--- 8 绩效
    ratio_ls = [fig_label(factor_return[col])[1] + [factor_turnover(dict_sig[col])] for col in [f'group_{i}' for i in list_col]]
    
    ratio_ls = [float("{:.4f}".format(item)) for sublist in ratio_ls for item in sublist]
    #--- 9 IC IR
    [IC, IC_mean, ICIR,  rankIC, rankIC_mean, rankICIR] = calc_icir_single(factor, ret)
    
    ratio_ic = [float("{:.4f}".format(metric)) for metric in [IC_mean, ICIR, rankIC_mean, rankICIR]]
    
    
    #--- 10 费后损益
    if cost is None:
        cost = 0
        
        ratio_ls_fee = [np.nan] * len(list_col) * 5    #绩效输出的数量    
        
    else:
        #! 注意，这里应该要除以持有期
        # cost = cost / h
        for col in [f'group_{i}' for i in list_col]:
            # 总持仓品种数量
            temp = dict_sig[col].abs().sum(axis=1).shift()
            # 交易损益（跳空收益-手续费）
            fee = (dict_sig[col].diff().shift() * cost).sum(axis=1).div(temp, axis=0)
            
            factor_return[f'{col}_fee'] = (dict_sig[col].shift() * rets).sum(axis=1).div(temp, axis=0) - fee
            
        ratio_ls_fee = [fig_label(factor_return[f'{col}_fee'])[1] for col in [f'group_{i}' for i in list_col]]
        
        ratio_ls_fee = [float("{:.4f}".format(item)) for sublist in ratio_ls_fee for item in sublist]
    
        
    #--- 11 因子绩效汇总
    # 15分组+24分组+1245分组+15费后+IC
    ratio = ratio_ls_fee + ratio_ic + ratio_ls 


    # #--- 12 因子累计收益
    factor_return = factor_return.fillna(0).cumsum()#.xs(factorName,level=1).reset_index(level=1,drop=True)
    
    
    #--- 13 plot 
    if fig:
        _,ax = plt.subplots(3,1)
        
        ax[0].plot(factor_return['group_LS_1'], 'r',label=f'sr={ratio[1]}')
        
        for col in list_col:
            ax[0].plot(factor_return[f'group_{col}_fee'] ,alpha=0.7, label=f'{col}_fee')
                    
        ax[0].legend(loc='best',fontsize='x-small')
        ax[0].grid(True)
        ax[0].set_title(factorName)

        for col in factor_return.columns[:groupNum]:
            ax[1].plot(factor_return[col],label=col,alpha=0.7)
            
        ax[1].legend(loc=2,fontsize='x-small')
        ax[1].grid(True)
 
        ax[2].plot(IC.cumsum(), label='IC')
        ax[2].plot(rankIC.cumsum(), label='rankIC')
        ax[2].legend(loc='best',fontsize='x-small')
        ax[2].grid(True)
        
        # plt.subplots_adjust(hspace=0)
        
        plt.tight_layout()
        
        _.savefig(fig_path)

        plt.close()
 
    return factor_return, dict_sig['group_LS_1'],  ratio


def compute_forward_returns(rets, periods=(1, 3, 5, 10,15,20)):
    
    return_forward = pd.DataFrame()
    
    for hp in periods:
        
        temp = rets.rolling(hp).mean().shift(-hp)
        
        temp['hp'] = f'{str(hp)}D'
        
        temp.set_index('hp', append=True, inplace=True)
        
        return_forward = pd.concat([return_forward, temp])
    return return_forward


def factor_test_single(factor: pd.DataFrame, rets: pd.DataFrame, forward_return: pd.DataFrame, 
                       cost: pd.DataFrame, groupNum:int, groupInd: int, hp:int):
    
        factor_rank = factor.rank(axis=1, ascending=False)
        
        factor_signal = (1 * grouper(factor_rank, groupNum, n=groupInd, method='ceil') \
                         - 1 * grouper(factor_rank, groupNum, n=(groupNum+1-groupInd), method='ceil'))
        
        pos = factor_signal.rolling(hp).sum() / hp
        
        if cost is None:
            cost = 0
            print('no fee')
        
        fee =  pos.fillna(0).diff().shift() * cost
        
        lots = pos.shift().abs().sum(axis=1)
        
        factor_return = (pos.shift() * rets - fee).sum(axis=1).div(lots, axis=0)
            
        # factors information coefficiency
        factor_ic = factor_signal.corrwith(forward_return,axis=1)

        return factor_return, factor_ic, pos


def factor_test_all(factors: pd.DataFrame, rets: pd.DataFrame, groupNum: int, 
                    groupInd=1, periods=(1, 3, 5, 10, 15, 20)):
    '''

    Parameters
    ----------
    factors : pd.DataFrame
        所有因子数据
    rets : DataFrame
        日收益率
    groupNum : int
        分组数量
    groupInd: int
        分组组别
    periods : iterable arrat, optional
        持有期列表. The default is (1, 3, 5, 10,15,20).

    Returns
    -------
    None.

    '''
    # factors groupby factorName
    factors_groupby = factors.groupby(level=1)
    
    # 持有期收益
    forward_returns = compute_forward_returns(rets, periods)
    
# =============================================================================
#     # category factors into groupNum groups
#     factors_group = factor_fast_groupon(factors, groupNum)
#     # long short signal
#     factors_signal = ( 1 * (factors_group == 1) - 1 * (factors_group == groupNum) )
#     
# =============================================================================
    
    # 另一种能让多空品种数量完全一致，且速度也比较快的方式
    factor_rank = factors.rank(axis=1, ascending=False)
    
    factors_signal = (1 * grouper(factor_rank, groupNum, n=groupInd, method='ceil') \
                     - 1 * grouper(factor_rank, groupNum, n=(groupNum+1-groupInd), method='ceil'))
    
    # singal groupby factorName
    # factors_signal_groupby = factors_signal.groupby(level=1) 
    factors_signal_groupby = factors_signal.groupby('factor') 
    
    factors_return_all = pd.DataFrame()
    
    factors_ic_all = pd.DataFrame()
    
    for hp in periods:
        
        str_hp = f'{hp}D'
        
        forward_return = forward_returns.xs(str_hp, level='hp')
        
        factors_return = factors_signal_groupby.apply(lambda x: \
                        ((x.rolling(hp).sum() / hp).shift().reset_index(level=1, drop=True) * rets).sum(axis=1).\
                        div((x.rolling(hp).sum() / hp).shift().reset_index(level=1, drop=True).abs().sum(axis=1), axis=0))
        
            
        factors_return['hp'] = str_hp
        
        factors_return.set_index('hp', append=True, inplace=True)
        
            
        # factors information coefficiency
        factors_ic = factors_groupby.apply(lambda x: \
                    x.reset_index(level=1,drop=True).corrwith(forward_return,axis=1))

        # factors_ic = factors_ic.T
        
        factors_ic['hp'] = str_hp 
        
        factors_ic.set_index('hp', append=True, inplace=True)
        
        # concate 
        factors_return_all = pd.concat([factors_return_all, factors_return])
        
        factors_ic_all = pd.concat([factors_ic_all, factors_ic])
        
    return factors_return_all, factors_ic_all, factors_signal






