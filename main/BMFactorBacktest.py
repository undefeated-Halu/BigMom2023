# -*- coding: utf-8 -*-
"""
strategyName:
    BigMom2023
edition: 
    2023Oct
strategyType:
    strategy_factor
Description: 
    日频多因子截面 基本面+量价  回测

TODOs:    

Created on Fri Dec 29 09:54:33 2023

Edited on Fri Dec 29 09:54:33 2023


@author: oOoOo_Andra
"""

root_path = 'D:/ProgramFiles/python/'

root_data = 'D:/Data/'

path_lib = f'{root_path}BASE/'

path_strategy = f'{root_path}strategy_factor/BigMom2023/'

import pandas as pd
import numpy as np
import datetime
import os
os.chdir(path_strategy)
#import time
import sys
sys.path.append('D:/Data/factorFunda/')
sys.path.append(path_lib)
sys.path.append(f'{path_strategy}main/')

from BigMomWTS import *
from FactorCalcFunctions import *
from FactorBaseFunctions import *
import yaml
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#%%
'''
这里会用一个yaml文件作为策略所有参数、路径等信息的配置文件
'''
with open('config/config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
    configFile = yaml.safe_load(parameters)

paramsBank = configFile['params']

pathBank = configFile['path']

pathTest = configFile['test']

locals().update(paramsBank['basic'])
locals().update(pathBank)
locals().update(pathTest)


### 1 基础信息表
future_info, cols_index, list_factor_test, df_factorTable = load_basic_info(filepath_future_list, filepath_factorTable2023)


### 2 品种日数据
dfindex, retIndex, costIndex, dfmain, retMain, costMain = load_local_data(filepath_index, filepath_main, future_info, start_date)

retIndex =  retIndex[cols_index]

print(f'index shape: {retIndex.shape}\n')
print(f'main shape: {retMain.shape}\n')

# 修正表头
cols_index = list(set(cols_index) & set(retMain.columns))

cols_index.sort()

# 修正长度(以main为基准)
if dailyReturnMode == 'main':
    
    if len(retMain) != len(retIndex):
        merged = pd.merge(retIndex[['A']], retMain[['A']], how='outer', left_index=True, right_index=True, suffixes=('_retIndex', '_retMain'))
        print(merged[merged.A_retIndex.isna()])
        
    '''
    收益日序列全用主力数据
    但是，基本面数据比较难补，
    还是以指数数据长度为准
    '''
    [dailyReturn_all, cost] = trimShape(retIndex[cols_index], 0, True,True, retMain, costMain)
    
    print('This factor test is based on main returns!!\n')

else:
    dailyReturn_all = retIndex
    cost = costIndex
    
    print('This factor test is based on index returns!!\n')

print(f'dailyReturn_all({dailyReturnMode}): {dailyReturn_all.shape}\n')
print(f'cost({dailyReturnMode}): {cost.shape}\n')



wts = WTS_factor(dfindex, cols_index)

### 6 结果保存路径
Description = dailyReturnMode

test_date = 'factorTest_2023Dec_'

# 输出文件夹
filepath_test_output = f'{filepath_output}{test_date}{Description}/'

filepath_output_ratios_all = f'{filepath_test_output}performance_ratios{Description}.csv'


#%%
# 分组测试累计日收益的结果
dfret = pd.DataFrame()
# 收益绩效统计表
dfratio = pd.DataFrame(columns=list_ratios)


'''    
factorName = 'alpha_206'
factorName = 'alpha_082'
factorName = 'alpha_f2'

'''
### 测试
for factorName in list_factor_test[229:]:
# for factorName in ['alpha_f1']:
   
    print(factorName)
    
    ### 参数组生产
    parameter_list, list_paramName, list_paramSpace = generate_paramList(factorName, df_factorTable)
    
    # 按照因子保存结果
    filepath_output_factor = f'{filepath_test_output}{factorName}/'
    
    if not os.path.exists(filepath_output_factor) :
        os.makedirs(filepath_output_factor)
    
    #单因子测试绩效
    dfratio_factor = pd.DataFrame(columns=list_ratios)
    
    ### 参数循环
    for param,i  in zip(parameter_list, range(len(parameter_list))):
             
        # 测试结果保存位置
        test_name = f'{factorName}_param{i}'
        
        filepath_fig = f'{filepath_output_factor}{test_name}'
        
        hp = param[-1]

        #---1 计算因子
        factor = eval(f'wts.{factorName}{param[:-1]}')
                      
        if factor is None:
            print(param, i ,' invalid parameters!')
            
        else:
        #---2 因子处理
            '''
            2.03 ms ± 43.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            '''            
            if factor.shape != dailyReturn_all.shape:
                [factor] = trimShape(retIndex, 'ffill', True, True, factor)
                
            [_ret, _cost] = trimShape(factor, 0 ,True, True, dailyReturn_all, cost)
            
            # 调用因子处理函数
            factor = WTS_factor_handle(factor, nsigma=3)
        #---3 截面测试
            # 有交易费的测试
            _, _, ratio = factor_test_group(factor, _ret, _cost, h=hp, factorName=test_name,
                                            fig=bool_test_fig, fig_path=filepath_fig)
            
            # dfratio.loc[test_name,:] = ratio + [factorName] +  [str(param)]
            dfratio_factor.loc[str(i),:] = [str(param)] + ratio + [factorName]
    
    ### 绩效保存
    # 单因子测试绩效保存
    dfratio_factor.to_csv(f'{filepath_output_factor}{factorName}_performance_ratios.csv')
    
    # 汇总合并
    dfratio = pd.concat([dfratio, dfratio_factor], axis=0)
    
    ### 因子绩效作图
    # 先把参数组拆开
    temp = dfratio_factor['parameter'].apply(lambda x : x.strip('()').split(', '))
    
    for i, param in zip(range(len(list_paramName)), list_paramName):
        
        # dfratio_factor[param] = temp.apply(lambda x: int(x[i]))
        dfratio_factor[param] = temp.apply(lambda x: eval(x[i]))
    
    
    fig, ax = plt.subplots(3,1)
    
    sns.stripplot(x='N', y='ar_15', hue='hp', data=dfratio_factor, jitter=0.25, size=8, ax=ax[0], linewidth=.5, palette='deep')
    
    ax[0].set_title(f'{factorName} performance scatter plot')
    
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    sns.stripplot(x='N', y='sr_15', hue='hp', data=dfratio_factor, jitter=0.25,  size=8, ax=ax[1], linewidth=.5, palette='deep', legend=False)
    
    sns.stripplot(x='N', y='mar_15', hue='hp', data=dfratio_factor, jitter=0.25, size=8, ax=ax[2], linewidth=.5, palette='deep', legend=False)
    
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    fig.savefig(f'{filepath_output_factor}{factorName}_ratio')
    
    plt.close()
    
# 单因子测试绩效保存   
dfratio.to_csv(f'{filepath_output_ratios_all}')

print(f'''{filepath_output_ratios_all}''')
