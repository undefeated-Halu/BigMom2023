# -*- coding: utf-8 -*-
"""
strategyName:
    BigMom2023 Factor Weight Test
edition: 
    2023Nov15
strategyType:
    strategy_factor
Description: 
    
TODO:    
    因子选择
    因子组合
    权重分配
    
Created on Tue Oct 24 17:33:02 2023
Edited on Fri Nov 17 21:33:00 2023

@author: oOoOo_Andra
"""

root_path = 'D:/ProgramFiles/python/strategy_factor/BigMom2023/'

root_data = 'D:/Data/'

import pandas as pd
import numpy as np
import os
# os.chdir(root_path)
import yaml
import sys
sys.path.append(f'{root_path}main')

import matplotlib.pyplot as plt
import BigMomWTS as bm
import FactorModelingFunctions as bmModel
import FactorBaseFunctions as bmBase
import scipy.optimize as sco
from datetime import datetime

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
import itertools

with open('config/config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
    configFile = yaml.safe_load(parameters)

paramsBank = configFile['params']

pathBank = configFile['path']

pathFactor_weight = configFile['factor_weight']

locals().update(paramsBank['basic'])
locals().update(pathBank)
locals().update(pathFactor_weight)

del configFile, paramsBank, pathBank, pathFactor_weight

#% laod basic
### 1 基础信息表
future_info, trade_cols, list_factor_test,df_factorTable = bmBase.load_basic_info(filepath_future_list, filepath_factorTable2023)

### 2 品种指数日数据
price, rets, retMain, cost = bmBase.load_local_data(filepath_index,filepath_factorsF,future_info, trade_cols, start_date, end_date)
# rets = retIndex[:end_date]

#--- 因子池文件
df_factorPools, periods = bmBase.load_factorPools(filepath_factorPools)

#--- forward return
forward_returns = bmBase.compute_forward_returns(rets, periods)

forward_returns_group = bmBase.factor_fast_groupon(forward_returns, groupNum=2)

#%% 生成 数据
if os.path.exists('data/factorsDailyRet.csv'):

    df_f_rets = pd.read_csv('data/factorsDailyRet.csv', index_col=0, parse_dates=True)
    df_f_signals = pd.read_csv('data/factorsDailySignal.csv', index_col=[0,1], parse_dates=True)
else:
    df_factorPools['timeConsumption'] = 0
    df_factorPools['tag_test'] = True
    
    #--- 生成
    wts = bm.WTS_factor(price, trade_cols)
    
    # factor returns without trade costs
    df_f_rets = pd.DataFrame(rets.index, columns=df_factorPools['testCode'])
    df_f_ic = pd.DataFrame(rets.index, columns=df_factorPools['testCode'])
    df_f_signals = pd.DataFrame()
    df_f = pd.DataFrame()
    
    
    for i in range(len(df_factorPools)):
            # 因子参数名称
        
        
        if ~df_factorPools.iloc[i, 0]:
            
            factorName = df_factorPools.index[i]
            
            try:
                paramSet = eval(df_factorPools.param[i])
        
                param = paramSet[:-1]
        
                hp = paramSet[-1]
                #---1 计算因子
                a = datetime.now()
                
                factor = eval(f'wts.{factorName}{param}')
                
                df_factorPools.iloc[i,-1]  = (datetime.now() - a).seconds
                
                
                #---2 因子处理
                if factor is None:
                    print(param, i ,' invalid parameters!')
                    df_factorPools.iloc[i, 0] = False
        
                else:
                    # 调用因子处理函数
                    print(factorName)
                    factor = bm.WTS_factor_handle(factor, nsigma=3)
                    # 预期收益
                    forward_return = forward_returns.xs(f'{hp}D', level='hp')
        
                #---3 单参数测试
                    df_f_rets[i], df_f_ic[i], factor_signal = bmBase.factor_test_single(factor, rets, forward_return, cost, groupNum=5, groupInd=1, hp=hp)
        
                    factor_signal['factor'] = i
        
        
                    df_f_signals = pd.concat([df_f_signals, factor_signal])
        
                    factor['factor'] = i
                    df_f = pd.concat([df_f, factor])
                    
            except:
                print(f'{factorName} is not valid')
                df_factorPools.iloc[i, 0] = False
    
    #--- trim data
    
    # 统一index格式
    df_f_rets.index = pd.to_datetime(df_f_rets.index, format='%Y-%m-%d')
    
    df_f_rets = df_f_rets.replace([np.inf,-np.inf], 0).fillna(0)
    
    df_f_ic.index = pd.to_datetime(df_f_ic.index, format='%Y-%m-%d')
    # df_f_signals.index.levels[0] = pd.to_datetime(factor_signal.index.levels[0], format='%Y-%m-%d')
    
    df_f_signals.set_index('factor', append=True, inplace=True)
    
    df_f.set_index('factor', append=True, inplace=True)
    
    list_features = list(np.arange(len(df_factorPools)))
    
    #--- 保存文件
    df_f_rets.to_csv('data/factorsDailyRet.csv')
    
    df_f_signals.to_csv('data/factorsDailySignal.csv')
    
    df_f.to_csv('data/factors.csv')
    
    df_factorPools.to_excel(filepath_factorPools)


#% 因子组合 1 样本外
dt = '2022-12-30'

n_prod = df_f_rets.shape[1]


#--- 0 因子等权
 
weight = pd.Series(1/n_prod, index=df_f_rets.columns)

signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)

for i in range(len(weight)):
    
    signal += df_f_signals.xs(i, level=1) * weight[i]

signal = signal.div(signal.abs().sum(axis=1), axis=0)

ret_ptf_avg = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()

print(ret_ptf_avg.tail())
#%% SCO样本外
#--- 1样本内外
#----- 1.1优化器优化

weight_max = 0.2

cons=({'type':'eq',
       'fun':lambda x:np.sum(x)-1})

bnds = tuple((0, weight_max) for x in range(n_prod))

opts = sco.minimize(fun = bmModel.max_sp ,
                    x0  = n_prod * [1./n_prod,] ,
                    method = 'SLSQP' ,
                    args = (df_f_rets[:dt],) ,
                    bounds = bnds ,
                    constraints = cons)

res = opts.x

weight = pd.DataFrame(res, index = df_f_rets.columns)

signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)

for i in range(len(weight)):
    
    signal += df_f_signals.xs(i, level=1) * weight.iloc[i,0]

signal = signal.div(signal.abs().sum(axis=1), axis=0)

ret_ = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()

_, ax = plt.subplots()
ax.plot(ret_ptf['avg'], 'g', label='average')
ax.plot(ret_ptf['avg'][dt:],  'r', lw=2)
# ax.plot(ret_ptf['icir'][:date_out], 'b', label='sr')
ax.plot(ret_, 'k', label='sco_fee')
ax.plot(ret_[dt:], 'r', lw=2, label='outSample')
ax.legend(loc=2)
ax.grid(True)

ret_ptf['outSample'] = ret_


#%% SCO 推进 
'''
下面这段代码运行速度超级慢，原因是SCO优化器很慢
别轻易跑这段代码！！！！



'''
sco_test = False#--- 推进
if sco_test:
    
    # =============================================================================
    # window_size = 250
    # 
    # step_size = 20
    # 
    # weight_max = 0.2
    # 
    # =============================================================================
    # 滚动回测
    
    
    list_ws = [120, 250, 500]
    
    list_step = [20, 60]
    
    list_maxW = [0.1, 0.2]
    
    parameter_list = list(itertools.product(list_ws,list_step,list_maxW))
    
    list_func = ['max_sp','max_sortino','min_volatility','max_upside_volatility','min_downside_volatility']
    
    dfweight = pd.DataFrame(np.NaN, index=rets.index, columns=df_f_rets.columns)
    
    n_day = dfweight.shape[0]
    
    ret_all = pd.DataFrame()
    perf_ratios_all = pd.DataFrame()
    
    for window_size, step_size, weight_max in parameter_list[1:]:
        
        test_code = '_'.join(str(s) for s in (window_size, step_size, int(weight_max*100)))
        print(test_code)
        print('$'*25)
    
        cons=({'type':'eq',
               'fun':lambda x: np.sum(x)-1})
        
        bnds = tuple((0, weight_max) for x in range(n_prod))
    
        for func in list_func:
            print(func)
        
            for i in range(window_size, n_day, step_size):
                
                print(df_f_rets.index[i])
                
                df_rets_slice = df_f_rets.iloc[i-window_size:i, ]
                
                opts = sco.minimize(
                                    fun = eval(f'bmModel.{func}'),
                                    x0  = n_prod * [1./n_prod,] ,
                                    method = 'SLSQP' ,
                                    args = (df_rets_slice ) ,
                                    bounds = bnds ,
                                    constraints = cons)
                
                res = opts.x
                    
                dfweight.iloc[i-1,:] = res
        
        
            # 填充nan值
            dfweightCopy = dfweight.fillna(method='ffill').fillna(1/dfweight.shape[1])
            
            signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)
            
            for i in range(len(res)):
                
                signal += df_f_signals.xs(i, level=1).mul(dfweightCopy.iloc[:,i], axis=0).fillna(0)
            
            signal = signal.div(signal.abs().sum(axis=1), axis=0)
            
            ret_ptf[func] = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()
    
        perf_ratios = pd.DataFrame(columns=ret_ptf.columns,index=['ar','sr','mdd','mar'])
        
        fig, ax = plt.subplots()
        
        # ax.plot(ret_ptf, 'k', label=f'back{window_size}forward{step_size}')
        for col in ['avg'] + list_func:
            perf_ratios[col] = bmBase.performance_ratio(ret_ptf[col].diff()[window_size:].fillna(0))
            
            ax.plot(ret_ptf[col][window_size:].diff().cumsum(), label=col, alpha=0.5)
            
        # ax.plot(ret_ptf['outSample'][window_size:].diff().cumsum()[dt:], 'r', lw=2)
        # ax.plot(ret_ptf['outSample'][dt:], 'r', lw=2, label='outSample')
        
        ax.set_title(f'back{window_size}forward{step_size}maxW{weight_max}')
        
        ax.legend(loc=2)
        
        ax.grid(True)
        
        
        fig.savefig(f'output/portfolio_param_test/performance_{test_code}')
        
        # print(perf_ratios )
    
        ret_ptf['test_code'] = test_code
        
        perf_ratios['test_code'] = test_code
    
        perf_ratios_all = pd.concat([perf_ratios_all, perf_ratios])
        perf_ratios.to_csv(f'output/portfolio_param_test/perf_ratios_{test_code}.csv')
        
        ret_all = pd.concat([ret_all, ret_ptf])
    
    
    ret_all.to_csv('output/portfolio_param_test/portfolio_param_test_cumret.csv')
    perf_ratios_all.to_csv('output/portfolio_param_test/perf_ratios_all.csv')    

# ret_ptf[['max_sp','min_volatility','min_downside_volatility']][window_size:].plot()


    

#%% cvx 样本外
import FactorModelingFunctions as bmModel

df_rets_slice = df_f_rets[:dt]

n = df_rets_slice.shape[1]

# weight = pd.DataFrame(bmModel.min_volatility_cvx(df_rets_slice, n, 0.1), index = df_f_rets.columns)
# weight = pd.DataFrame(bmModel.min_downside_volatility_cvx(df_rets_slice, n, 0.1), index = df_f_rets.columns)
# weight = pd.DataFrame(bmModel.max_adjustedReturn_cvx(df_rets_slice, n, 0.1), index = df_f_rets.columns)
weight = pd.DataFrame(bmModel.max_upside_volatility_cvx(df_rets_slice, n, 0.1), index = df_f_rets.columns)


signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)

for i in range(len(weight)):
    
    signal += df_f_signals.xs(i, level=1) * weight.iloc[i,0]

signal = signal.div(signal.abs().sum(axis=1), axis=0)

ret_ = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()

_, ax = plt.subplots()
ax.plot(ret_ptf['avg'], 'k', label='average')
ax.plot(ret_ptf['avg'][dt:],  'r', lw=2)
# ax.plot(ret_ptf['icir'][:date_out], 'b', label='sr')
ax.plot(ret_, 'b', label='cvx_fee')
ax.plot(ret_[dt:], 'r', lw=2, label='outSample')
ax.legend(loc=2)
ax.grid(True)


plt.figure()
weight.plot()


#%% cvx 推进

'''
window_size = 120

step_size = 20

weight_max = 0.1

dfweight = pd.DataFrame(np.NaN, index=rets.index, columns=df_f_rets.columns)'
'''

n_day = df_f_rets.shape[0]

list_ws = [120, 250, 500]

list_step = [20, 60]

list_maxW = [0.05, 0.1, 0.2]

parameter_list = list(itertools.product(list_ws,list_step,list_maxW))

list_func = ['max_adjustedReturn_cvx','min_volatility_cvx','min_downside_volatility_cvx']

dfweight = pd.DataFrame(np.NaN, index=rets.index, columns=df_f_rets.columns)

n_day = dfweight.shape[0]

ret_all = pd.DataFrame()
perf_ratios_all = pd.DataFrame()

for window_size, step_size, weight_max in parameter_list[1:]:
    
    test_code = '_'.join(str(s) for s in (window_size, step_size, int(weight_max*100)))
    print(test_code)
    print('$'*25)
    
    ret_ptf = ret_ptf_avg.to_frame('avg')
    
    for func in list_func:
        print(func)
        print('')

        for i in range(window_size, n_day, step_size):
            
            # print(df_f_rets.index[i])
            
            df_rets_slice = df_f_rets.iloc[i-window_size:i, ]
            
            n = df_rets_slice.shape[1]
                
            # dfweight.iloc[i-1,:] = bmModel.min_volatility_cvx(df_rets_slice, n, weight_max)
            try:
                dfweight.iloc[i-1,:] = eval(f'bmModel.{func}(df_rets_slice, n, weight_max)')
            except:
                print(df_f_rets.index[i])
                dfweight.iloc[i-1,:] = 1 / n
        
        # 填充nan值
        dfweightCopy = dfweight.fillna(method='ffill').fillna(1/dfweight.shape[1])
        
        signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)
        
        for i in range(df_f_rets.shape[1]):
            
            signal += df_f_signals.xs(i, level=1).mul(dfweightCopy.iloc[:,i], axis=0).fillna(0)
        
        signal = signal.div(signal.abs().sum(axis=1), axis=0)
        
        ret_ptf[func] = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()
    
        perf_ratios = pd.DataFrame(columns=ret_ptf.columns, index=['ar','sr','mdd','mar'])
        
    fig, ax = plt.subplots()
    
    # ax.plot(ret_ptf, 'k', label=f'back{window_size}forward{step_size}')
    for col in ret_ptf.columns:
        
        perf_ratios[col] = bmBase.performance_ratio(ret_ptf[col].diff()[window_size:].fillna(0))
        
        ax.plot(ret_ptf[col][window_size:].diff().cumsum(), label=col, alpha=0.5)
        
    
    ax.set_title(f'CVX-back{window_size}forward{step_size}maxW{weight_max}')
    
    ax.legend(loc=2)
    
    ax.grid(True)
    
    
    fig.savefig(f'output/portfolio_param_testCVX/performance_{test_code}')
        
    print(perf_ratios )
    
    ret_ptf['test_code'] = test_code
    
    perf_ratios['test_code'] = test_code

    perf_ratios_all = pd.concat([perf_ratios_all, perf_ratios])
    
    ret_all = pd.concat([ret_all, ret_ptf])
    
    
ret_all.to_csv('output/portfolio_param_testCVX/portfolio_param_test_cumret.csv')
perf_ratios_all.to_csv('output/portfolio_param_testCVX/perf_ratios_all.csv')    

'''
_, ax = plt.subplots()
ax.plot(ret_ptf['avg'], 'k', label='average')
ax.plot(ret_ptf['avg'][dt:],  'r-', lw=1)
# ax.plot(ret_ptf['icir'][:date_out], 'b', label='sr')
ax.plot(ret_, 'b', label='cvx_fee')
ax.plot(ret_[dt:], 'r', lw=2, label='outSample')
ax.legend(loc=2)
ax.grid(True)
'''








