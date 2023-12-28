# -*- coding: utf-8 -*-
"""
Fundamental factor generating script.

Datasets are based on basci files from D:/Data/tushare


strategyName:
    BigMom2023
edition: 
    2023Dec
strategyType:
    strategy_factor
Description: 
    日频多因子截面，基本面

Created on Wed Dec 27 12:48:03 2023

Edited on Wed Dec 27 12:48:03 2023

@author: oOoOo_Andra
"""

import pandas as pd
import numpy as np
import datetime
import time
import sys
sys.path.append('D:/ProgramFiles/python/strategy_factor/BigMom2023/main/')
from FactorBaseFunctions import *



def fundaFactorGenerator():

    #--- path
    root_path = 'D:/ProgramFiles/python/'
    
    root_data = 'D:/Data/tushare/'
    
    path_factor = f'{root_data}factor/'

    path_data = path_factor
    
    start_date='2016-01-01'
    
    #--- main contract
    filepath = f'{root_data}quote/quote2.csv'
    
    df_MS_sp = pd.read_csv(filepath, index_col='date',parse_dates=True)[start_date:]


    # main spreadsheet( whole dataset)
    df_main_all = df_MS_sp[df_MS_sp.type == 'main']
    df_main_all.loc[:,'pctChg'] = df_main_all.close / df_main_all.pre_close - 1
    df_main_all.loc[:,'jump'] = df_main_all.open / df_main_all.pre_close - 1
    
    
    # main close price
    df_main = df_main_all[['product','close']].set_index(['product'],append=True).unstack(level=1)
    df_main = df_main.droplevel(level=0, axis='columns')
    df_main.sort_index(inplace=True)
    
    df_main.to_csv(f'{path_data}main.csv')
    
    # main return
    df_main_ret = df_main_all[['product','pctChg']].set_index(['product'],append=True).unstack(level=1)
    df_main_ret = df_main_ret.droplevel(level=0, axis='columns')
    df_main_ret.sort_index(inplace=True)
    
    df_main_ret.to_csv(f'{path_data}retMain.csv')
    
    # main jump return
    df_main_jump = df_main_all[['product','jump']].set_index(['product'],append=True).unstack(level=1)
    df_main_jump = df_main_jump.droplevel(level=0, axis='columns')
    df_main_jump.sort_index(inplace=True)
    
    df_main_jump.to_csv(f'{path_data}retMainJump.csv')
    
    # main contract date number
    df_main_dt = df_main_all[['product','contract_month']].set_index(['product'],append=True).unstack(level=1)
    df_main_dt = df_main_dt.droplevel(level=0, axis='columns')
    df_main_dt.sort_index(inplace=True)
    
    
    #--- main sub spread
    filepath = f'{path_factor}rolling_return2.csv'
    
    df_main_sub = pd.read_csv(filepath, index_col='date', parse_dates=True)[start_date:] 
   
    df_main_sub.to_csv(f'{path_data}main_sub_spread.csv')
    
    #--- spot_basis
    filepath =  f'{path_factor}spot_price.csv'
    
    df_spot_price = pd.read_csv(filepath,index_col=0, parse_dates=True)[start_date: ]
    
    df_spot_price.sort_index(inplace=True)
    
    df_spot_price = df_spot_price.replace(0, np.nan)
    
    df_spot_price.fillna(method='ffill', inplace=True)
    
    # df_spot_price = df_spot_price.drop(columns=['CY', 'PM', 'WH','RS','WR'])
    
    df_spot_price = trim_factor(df_spot_price,df_main)[0]
    
    cols = df_spot_price.columns.tolist()
    """
    应该是用交割日来倒推，但是交割日数据太麻烦。取一个大概的数字
    """
    # 交割日都按照每个月30号算,每个月30天
    df_spot_dt = df_main_dt[cols] * 30
   
    trade_date = df_spot_dt.index.strftime('%y%m%d').astype('int').to_series()

    trade_date = (trade_date // 10000 * 12 + trade_date % 10000 // 100) * 30 + trade_date % 10000 % 100
    
    df_spot_dt = df_spot_dt.apply(lambda x: x - trade_date.values)
    
    df_spot_basis = (df_main[cols] / df_spot_price[cols] - 1).fillna(method='ffill')
    
    df_spot_basis.to_csv(f'{path_data}spot_basis.csv')

    #--- OI

    df_long = pd.read_csv(f'{path_factor}fut_long.csv',index_col=0, parse_dates=True,).fillna(method='ffill')
    
    df_short = pd.read_csv(f'{path_factor}fut_short.csv',index_col=0, parse_dates=True,).fillna(method='ffill')

    df_all = (df_long + df_short).replace(0, np.nan).fillna(method='ffill')[start_date: ]
    
    df_all.to_csv(f'{path_data}oi_all.csv')
    
    df_spot_basis.to_csv(f'{path_data}oi_all.csv')
    
    df_raw = (df_long - df_short).fillna(method='ffill')[start_date: ]
    
    df_raw.to_csv(f'{path_data}raw_ls.csv')
    
    











