# -*- coding: utf-8 -*-
"""
strategyName:
    BigMom2023 Factor Weight
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

root_path = 'D:/ProgramFiles/python/'

root_data = 'D:/Data/'

path_lib = f'{root_path}BASE/'

path_strategy = f'{root_path}strategy_factor/BigMom2023/'

import pandas as pd
import numpy as np
import os
os.chdir(path_strategy)
#import time
import sys
sys.path.append('D:/Data/factorFunda/')
sys.path.append(path_lib)
# sys.path.append(path_strategy)
import yaml
# import ctaBasicFunc as cta
# from datetime import datetime
import matplotlib.pyplot as plt
import BigMomWTS_2309 as bm
import FactorModelingFunctions as bmModel
import FactorBaseFunctions as bmBase
import scipy.optimize as sco
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

with open('config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
    configFile = yaml.safe_load(parameters)

paramsBank = configFile['params']

pathBank = configFile['path']

pathFactor_weight = configFile['factor_weight']

locals().update(paramsBank['basic'])
locals().update(pathBank)
locals().update(pathFactor_weight)

del configFile, paramsBank, pathBank, pathFactor_weight

#%% laod basic
### 1 基础信息表
test_date = 'factorTest_23Nov17'

Description = ''

future_info, trade_cols, list_factor_test,df_factorTable = bm.load_basic_info(filepath_future_list, filepath_factorTable2023)

### 2-5 品种指数日数据
price, rets, retMain, cost = bm.load_local_data(filepath_index,filepath_factorsF,future_info, trade_cols, start_date, end_date)
# rets = retIndex[:end_date]


#--- 因子池文件已经
df_factorPools = pd.read_excel(filepath_factorPools, index_col=0)

df_factorPools['testCode'] = range(len(df_factorPools))

df_factorPools['hp'] = df_factorPools.param.apply(lambda x: eval(x)[-1])

periods = df_factorPools['hp'].unique()

print(f'''factorPools长度： {df_factorPools.shape[0]}''')

#--- forward return
forward_returns = bmBase.compute_forward_returns(rets, periods)

#%% 生成 数据
#--- 生成
wts = bm.WTS_factor(price, trade_cols)

# factor returns without trade costs
df_f_rets = pd.DataFrame(rets.index, columns=df_factorPools['testCode'])
df_f_ic = pd.DataFrame(rets.index, columns=df_factorPools['testCode'])
df_f_signals = pd.DataFrame()


for i in range(len(df_factorPools)):
    # 因子参数名称
    factorName = df_factorPools.index[i]
    
    paramSet = eval(df_factorPools.param[i])
    
    param = paramSet[:-1]
    
    hp = paramSet[-1]
    
    print(factorName)
    
    #---1 计算因子
    factor = eval(f'wts.{factorName}{param}')
    
    #---2 因子处理
                 
    if factor is None:
        print(param, i ,' invalid parameters!')   
        
    else:
        # 调用因子处理函数
        factor = bm.WTS_factor_handle(factor, nsigma=3)
        # 预期收益
        forward_return = forward_returns.xs(f'{hp}D', level='hp')
        
    #---3 单参数测试
        df_f_rets[i], df_f_ic[i], factor_signal = bmBase.factor_test_single(factor, rets, forward_return, cost, groupNum=5, groupInd=1, hp=hp)
        
        factor_signal['factor'] = i
        
        factor_signal.set_index('factor', append=True, inplace=True)
        
        df_f_signals = pd.concat([df_f_signals, factor_signal])

# 统一index格式
df_f_rets.index = pd.to_datetime(df_f_rets.index, format='%Y-%m-%d')
df_f_ic.index = pd.to_datetime(df_f_ic.index, format='%Y-%m-%d')
factor_signal.index = pd.to_datetime(factor_signal.index, format='%Y-%m-%d')


#%% 因子组合
#--- 1样本内外
#----- 1.1优化器优化
dt = '2021'

n_prod = df_f_rets.shape[1]

weight_max = 0.2

cons=({'type':'eq',
       'fun':lambda x:np.sum(x)-1})

bnds = tuple((0, weight_max) for x in range(n_prod))

opts = sco.minimize(fun = bmModel.min_sp ,
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

ret_ptf = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()

_, ax = plt.subplots()
# ax.plot(ret_ptf['average'][:date_out], 'g', label='average')
# ax.plot(ret_ptf['icir'][:date_out], 'b', label='sr')
ax.plot(ret_ptf, 'k', label='sco_fee')
ax.plot(ret_ptf[dt:], 'r', lw=2, label='outSample')
ax.legend(loc=2)
ax.grid(True)

#--- 推进

dfweight = pd.DataFrame(np.NaN, index=rets.index, columns=df_f_rets.columns)

# 滚动回测
window_size = 250

step_size = 60

n_day = dfweight.shape[0]

for i in range(window_size, n_day, step_size):
    print(df_f_rets.index[i])
    
    df_rets_slice = df_f_rets.iloc[i-window_size:i,]
    
    opts = sco.minimize(fun = bmModel.min_sp ,
                        x0  = n_prod * [1./n_prod,] ,
                        method = 'SLSQP' ,
                        args = (df_rets_slice,) ,
                        bounds = bnds ,
                        constraints = cons)
    
    res = opts.x
        
    dfweight.iloc[i,:] = res


# 填充nan值
dfweightCopy = dfweight.fillna(method='ffill').fillna(1/dfweight.shape[1])

signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)

for i in range(len(weight)):
    
    signal += df_f_signals.xs(i, level=1).mul(dfweightCopy.iloc[:,i], axis=0).fillna(0)

signal = signal.div(signal.abs().sum(axis=1), axis=0)


ret_ptf = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()

_, ax = plt.subplots()

ax.plot(ret_ptf, 'k', label=f'back{window_size}forward{step_size}')

ax.plot(ret_ptf.iloc[:window_size,], 'g', label='average')

ax.legend(loc=2)

ax.grid(True)

#%%










