# -*- coding: utf-8 -*-
"""


strategyName:
    BigMom2023
edition: 
    2023Oct
strategyType:
    strategy_factor
Description: 
    日频多因子截面，基本面+量价
TODOs:
    1 数据构成
    2 因子构成
    3 因子测试
  
    

Created on Wed Sep 20 14:39:06 2023

Edited on Wed Sep 20 14:39:06 2023


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

from FactorCalcFunctions import *
from FactorBaseFunctions import *
import yaml
import ctaBasicFunc as cta
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



#%% 因子

class WTS_factor:
    def __init__(self, price, symbol_list):
        # benchmark_price = get_price(index, None, end_date, '1d',['open','close','low','high','avg_price','prev_close','volume'], False, None, 250,is_panel=1)
        self.OPEN       = price.loc[(slice(None),'open'),:].droplevel(1)[symbol_list]
        self.CLOSE      = price.loc[(slice(None),'close'),:].droplevel(1)[symbol_list]
        self.LOW        = price.loc[(slice(None),'low'),:].droplevel(1)[symbol_list]
        self.HIGH       = price.loc[(slice(None),'high'),:].droplevel(1)[symbol_list]
        # volume weighted average price
        self.VWAP       = price.loc[(slice(None),'avg'),:].droplevel(1)[symbol_list]
        # market value
        self.MKTV       = price.loc[(slice(None),'mkt_value'),:].droplevel(1)[symbol_list]
        # trade fees
        self.COST       = price.loc[(slice(None),'cost'),:].droplevel(1)[symbol_list]
        # market_weight
        self.MKTW = self.MKTV.div(self.MKTV.sum(axis=1), axis=0)
        self.PRE_CLOSE  = price.loc[(slice(None),'pre_close'),:].droplevel(1)[symbol_list]
        self.VOLUME     = price.loc[(slice(None),'volume'),:].droplevel(1)[symbol_list]
        self.AMOUNT     = price.loc[(slice(None),'amount'),:].droplevel(1)[symbol_list]
        self.RET        = price.loc[(slice(None),'close'),:].droplevel(1).pct_change().fillna(0)[symbol_list]
        # self.HD         = self.HIGH - DELAY(self.HIGH,1)
        # self.LD         = DELAY(self.LOW,1) - self.LOW 
        # self.TR         = MAX(MAX(self.HIGH-self.LOW, ABS(self.HIGH-DELAY(self.CLOSE,1))),ABS(self.LOW-DELAY(self.CLOSE,1)))
        
    ### 因子
    '''
    206- 207: MOM
     / : RSM
    208-210 : RSI
    211-213 : OverNight 
    214-216 : Intraday 
    217-219 : MAratio
    220-222 : MAcross
    223-224 : CumStep
    225-227 : STDS
    228-229 : Rank
    230 : vol_GK
    231 : vol_RS
    232 : vol_PK

    '''
    
    def alpha_206(self,N=10):
        return -(self.CLOSE / DELAY(self.CLOSE,N) - 1)
    
    def alpha_207(self,N=20):
        return -(self.CLOSE / DELAY(self.CLOSE,N) - 1)
    
    def alpha_208(self,N=10):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET>0, self.RET, 0), N) / SUM(self.RET, N)

    def alpha_209(self,N=30):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET>0,self.RET, 0), N) / SUM(self.RET, N)
        
    def alpha_210(self,N=100):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET>0,self.RET, 0), N) / SUM(self.RET, N)
   
    def alpha_211(self, N=10):
        # 隔夜趋势因子
        return  MEAN(self.OPEN / self.PRE_CLOSE - 1, N)
    
    def alpha_212(self, N=30):
        # 隔夜趋势因子
        return  MEAN(self.OPEN / self.PRE_CLOSE - 1, N)
    
    def alpha_213(self, N=100):
        # 隔夜趋势因子
        return  MEAN(self.OPEN / self.PRE_CLOSE - 1, N)

    def alpha_214(self, N=10):
        # 日内动量因子
        return -1 * MEAN(self.CLOSE / self.OPEN - 1, N)
    
    def alpha_215(self, N=30):
        # 日内动量因子
        return -1 * MEAN(self.CLOSE / self.OPEN - 1, N)

    def alpha_216(self, N=100):
        # 日内动量因子
        return -1 * MEAN(self.CLOSE / self.OPEN - 1, N)

    def alpha_217(self, N=10):
        # k线均线因子(均线乖离率)
        return -1 * self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_218(self, N=30):
        # k线均线因子
        return -1 * self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_219(self, N=100):
        # k线均线因子
        return -1 * self.CLOSE / MEAN(self.CLOSE, N)

    def alpha_220(self, N=10):
        # 快慢均线趋势因子
        return -1 * MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1
    
    def alpha_221(self, N=30):
        # 快慢均线趋势因子
        return -1 * MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1
    
    def alpha_222(self, N=50):
        # 快慢均线趋势因子
        return -1 * MEAN(self.CLOSE,N) / MEAN(self.CLOSE,int(2 * N)) - 1

    def alpha_223(self, N=10):
        # 日内累计振幅因子
        # MEAN((2 * (HIGH - LOW) * SIGN(CLOSE - OPEN) - (CLOSE - OPEN)) / CLOSE, N)
        return-1 *  MEAN((2 * (self.HIGH - self.LOW) * SIGN(self.CLOSE - self.OPEN) 
                     - (self.CLOSE - self.OPEN)) / self.CLOSE, N)

    def alpha_224(self, N=30):
        # 日内累计振幅因子
        return -1 * MEAN((2 * (self.HIGH - self.LOW) * SIGN(self.CLOSE - self.OPEN) 
                     - (self.CLOSE - self.OPEN)) / self.CLOSE, N)

    def alpha_225(self, N=10):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return -1 *  MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)


    def alpha_226(self, N=40):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return  -1 * MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

    def alpha_227(self, N=100):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c**2)), N)) ** 0.5
        return -1 *  MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

        
    def alpha_228(self, N=10):
        # 稳健动量因子
        temp = RANK(self.RET,pct=False)
        n = temp.shape[1]      
        return -1 * MEAN( (temp - (n +1)/2) / (((n+1)*(n-1)/12)**0.5), N)

    def alpha_229(self, N=40):
        # 稳健动量因子
        temp = RANK(self.RET,pct=False)
        n = temp.shape[1]      
        return -1 * MEAN( (temp - (n +1)/2) / (((n+1)*(n-1)/12)**0.5), N)


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
        return (252 / 4 / np.log(2) * MEAN( (h - l) ** 2, N)) ** 0.5
    
    def alpha_238(self, N=20, M=40):
        # M>N
        if M > N:
            return self.CLOSE / DELAY(self.CLOSE, N) - self.CLOSE / DELAY(self.CLOSE, M)
        else:
            return None

    def alpha_901(self,kdj_len,fast_ema_len,slow_ema_len):   
        ll = bn.move_min(self.CLOSE,kdj_len,axis=0)
        hh = bn.move_max(self.CLOSE,kdj_len,axis=0)
        rsv = 100*(self.CLOSE - ll)/(hh-ll)
        k = rsv.ewm(alpha=1/fast_ema_len, adjust=False).mean()
        d = k.ewm(alpha=1/slow_ema_len, adjust=False).mean()
        j = k*3-d*2
        return -j

        ll = bn.move_min(CLOSE,kdj_len,axis=0)
        hh = bn.move_max(CLOSE,kdj_len,axis=0)
        rsv = 100*(CLOSE - ll)/(hh-ll)
        k = rsv.ewm(alpha=1/fast_ema_len, adjust=False).mean()
        d = k.ewm(alpha=1/slow_ema_len, adjust=False).mean()
        j = k*3-d*2

def WTS_factor_handle(factor, nsigma=3):
    #-----2.1 标准化（正态）
    factor = filter_extreme_normalize(factor, axis='columns', method=2)
    #-----2.2 处理缺失值
    factor = factor.fillna(method='ffill')
    #-----2.3 处理异常值（3sigma）
    factor[factor > nsigma] = nsigma 
    factor[factor < -nsigma] = -nsigma
    return factor

def load_basic_info(filepath_future_list, filepath_factorTable2023):
    ### 1 基础信息表
    # 期货列表，基础信息表
    future_info = pd.read_csv(filepath_future_list,index_col=0)
    
    # F和PV都交易的品种，基础信息表里用tradeF列标识
    trade_cols = future_info[future_info.tradeF.fillna(False)].index.tolist()
    
    # BigMom2023的factorTable
    df_factorTable = pd.read_excel(filepath_factorTable2023, index_col=0)
    
    list_factor_test = df_factorTable[df_factorTable.tag_test == 1].index.tolist()
    
    return future_info, trade_cols, list_factor_test, df_factorTable

def load_local_data(filepath_index, filepath_factorsF, future_info, 
                    trade_cols,start_date, end_date):
    ### 2 品种指数日数据
    print('load loca data......')
    print(f'start_date: {start_date}')
    print(f'end_date: {end_date}')
    # 品种指数数据，根据本地落地的1分钟数据生成的日数据，按照时间串行存储，通过code字段标记
    '''
                      open       high        low  ...  barsID    pctChg  code
    tradingday                                   ...                        
    2012-05-10   6190.000   6193.000   6082.000  ...       0  0.000000    AG
    2012-05-11   6110.000   6114.000   6001.000  ...       1 -0.022476    AG
    2012-05-14   6040.000   6045.000   5960.000  ...       2 -0.004332    AG
    '''
    price = pd.read_csv(filepath_index, index_col=0, parse_dates=True)
    
    price.sort_index(inplace=True)
    
    price = price[start_date : end_date]
    
    # 交易成本
    # 把品种乘数先当到表里
    price = price.merge(future_info.loc[trade_cols, ['point','type','commission']], left_on='code', right_index=True)
    # 手续费
    price['fee'] = np.where(price['type']==0, price['commission']/price.close/price.point,
                             price['commission'])
    # 总交易成本（跳空损益+手续费）
    price['cost'] = price['open'] / price['pre_close'] - 1 + price['fee']
    # 通过vwap和volume计算amount
    price['amount'] = price.avg * price.volume
    # 计算市值
    price['mkt_value'] = price.close * price.oi * price.point
    ## 后面几步是为了和之前的数据结构统一
    price.set_index('code',append=True, inplace=True)
    # 先把品种放回到列上，然后再把列上olhc字段放到index上。实现较为清晰的数据结构
    price = price.unstack(1).stack(0) 
    '''
    code                             A            AG  ...  ZC            ZN
    tradingday                                        ...                  
    2010-01-04 amount     1.309526e+09           NaN  ... NaN  8.155711e+09
               avg        4.068899e+03           NaN  ... NaN  2.150779e+04
               barsID     0.000000e+00           NaN  ... NaN  0.000000e+00
               close      4.057000e+03           NaN  ... NaN  2.143500e+04
               high       4.089000e+03           NaN  ... NaN  2.172000e+04
                               ...           ...  ...  ..           ...
    2023-07-13 oi         1.972420e+05  8.639010e+05  ... NaN  2.066940e+05
    '''
    
    ### 3 品种主力数据
    
    #--- 3.1 通过原quote2生成
    # =============================================================================
    # df_MS_sp = pd.read_csv(filepath_mainsub, index_col='date',parse_dates=True)[start_date:end_date]
    # df_main_all = df_MS_sp[df_MS_sp.type == 'main']
    # df_main_all.loc[:,'pctChg'] = df_main_all.close / df_main_all.pre_close - 1
    # df_main_all.loc[:,'jump'] = df_main_all.open / df_main_all.pre_close - 1
    # 
    # df_main_ret = df_main_all[['product','pctChg']].set_index(['product'],append=True).unstack(level=1)
    # df_main_ret = df_main_ret.droplevel(level=0, axis='columns')
    # df_main_ret.sort_index(inplace=True)
    # 
    # =============================================================================
    #--- 3.2 直接从h5文件读取
    df_main_ret = pd.read_hdf(filepath_factorsF, key='returns')[trade_cols][start_date : end_date]
    
    ### 4 生成实例
    wts = WTS_factor(price, trade_cols)
    
    ### 3 品种指数收益和主力合约收益
    retIndex = wts.CLOSE.pct_change().fillna(0)
    
    retMain = df_main_ret
    
    ### 5手续费
    cost = wts.COST.shift(-1).fillna(0)
    
    # 以指数长度为基准对齐主力收益
    temp = pd.DataFrame(1, index=retIndex.index, columns=['temp'])
    retMain = pd.merge(temp, retMain, how='left', left_index=True, right_index=True)[trade_cols].fillna(0)
    del temp
        
    return price, retIndex, retMain, cost

#%% MAIN
if __name__ == "__main__":

    '''
    这里会用一个yaml文件作为策略所有参数、路径等信息的配置文件
    '''
    with open('config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
        configFile = yaml.safe_load(parameters)
    
    paramsBank = configFile['params']
    
    pathBank = configFile['path']
    
    pathTest = configFile['test']
    
    locals().update(paramsBank['basic'])
    locals().update(pathBank)
    locals().update(pathTest)
    
    #---  准备工作
    ### 1 基础信息表
    future_info, trade_cols, list_factor_test,df_factorTable = load_basic_info(filepath_future_list, filepath_factorTable2023)
    
    ### 2-5 品种指数日数据
    # price, retIndex, retMain, cost = load_local_data(filepath_index, filepath_factorsF, future_info, start_date, end_date)
    price, retIndex, retMain, cost =  load_local_data(filepath_index, filepath_factorsF, future_info, 
                    trade_cols,start_date, end_date)
    
    
    wts = WTS_factor(price, trade_cols)
    ### 6 结果保存路径
    # sample_range = f'sample{start_date[:4]}to{end_date}'
    Description = ''
    
    test_date = 'factorTest_20231114'
    # test_date = f'''factorTest_{datetime.today().strftime("%Y%m%d")}'''
    
    # 输出文件夹
    filepath_test_output = f'{filepath_output}{test_date}{Description}/'
    
    filepath_output_ratios_all = f'{filepath_test_output}performance_ratios{Description}.csv'
    
    #--- 测试
    '''
    这里用的是全样本测试，
    示例是用指数收益
    '''
    ret = retIndex
    
    # 保证收益率和因子的表头一致
    # 因为retIndex是由CLASS wts生产的，所以是一样的
    if ret is retMain:
        if ret.columns is not retIndex.columns:
            
            tradeCols = list(set(ret.columns) & set(retIndex.columns))
            
            tradeCols.sort()
            
            wts = WTS_factor(price, tradeCols)
            
            ret = retMain[tradeCols]
            
        else:
            pass
    else:
        print('#'*25)
        print('This factor test is based on index returns!!')
        print('#'*25)
    
    
    # 分组测试累计日收益的结果
    dfret = pd.DataFrame()
    # 收益绩效统计表
    dfratio = pd.DataFrame(columns=list_ratios)
    
    
    '''
    # demo test
    wts = WTS_factor(price, trade_cols)
    
    i = 206
    hp = 5
    N = 60
    
    factorName = 'alpha_206'
    factorName = 'alpha_238'
    '''
    
        
    for factorName in list_factor_test:
    # for factorName in ['alpha_206']:
       
        print(factorName)
        
        '''
        在factorTable中， paramName 和 paramSpace 单元格内数据通过；来隔断
        
        load到脚本中为字符串的形式
        paramName:
        In : df_factorTable.loc[factorName, 'paramName'].split(';')
        Out: ['N', 'M', 'hp']
        
        相对应每个变量的变量空间也是字符串，但是要声明变量的类型，如range,list
        paramSpace:
            
        In： df_factorTable.loc[factorName, 'paramSpace'].split(';')
        Out: ['range(10,110,10)', 'range(10,110,20)', 'list((5,10))']   
        '''
        # load因子参数名称
        list_paramName = df_factorTable.loc[factorName, 'paramName'].split(';')
        # load因子参数空间（字符串） 
        list_paramSpace = df_factorTable.loc[factorName, 'paramSpace'].split(';')
        # 把因子从字符串形式eval成相对于的类型
        parameters = [eval(param) for param in list_paramSpace]
        # 生成参数列表
        parameter_list = list(itertools.product(*parameters))
        
        # 按照因子保存结果
        filepath_output_factor = f'{filepath_test_output}{factorName}/'
        
        if not os.path.exists(filepath_output_factor) :
            os.makedirs(filepath_output_factor)
        
        #单因子测试绩效
        dfratio_factor = pd.DataFrame(columns=list_ratios)
        
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
                # 调用因子处理函数
                factor = WTS_factor_handle(factor, nsigma=3)
            #---3 截面测试
                # 有交易费的测试
                _, _, ratio = factor_test_group(factor, ret, cost, h=hp, factorName=test_name,
                                                fig=bool_test_fig, fig_path=filepath_fig)
                
                # dfratio.loc[test_name,:] = ratio + [factorName] +  [str(param)]
                dfratio_factor.loc[str(i),:] = [str(param)] + ratio + [factorName]
        
        # 单因子测试绩效保存
        dfratio_factor.to_csv(f'{filepath_output_factor}{factorName}_performance_ratios.csv')
        # 汇总合并
        dfratio = pd.concat([dfratio, dfratio_factor], axis=0)
        
        #---4 因子绩效作图
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
    
    print(f'''
          Performance ratios excel sheet saved 
          @ {filepath_output_ratios_all}''')



