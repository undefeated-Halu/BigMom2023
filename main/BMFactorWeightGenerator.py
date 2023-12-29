# -*- coding: utf-8 -*-
"""


strategyName:
    BigMom2023 Factor Weight Generator
edition: 
    2023Dec11
strategyType:
    strategy_factor
Description: 
    
Created on Mon Dec 11 13:41:45 2023

Edited on Mon Dec 11 13:41:45 2023

@author: oOoOo_Andra
"""

# receiverList = ['pinglv08@163.com','wang_jun1016@163.com']

root_path = 'D:/ProgramFiles/python/strategy_factor/BigMom2023/'

root_data = 'D:/Data/'

import pandas as pd
import numpy as np
import os
os.chdir(root_path)
import yaml
import sys
sys.path.append(f'{root_path}main')
sys.path.append('D:/ProgramFiles/python/')

import BASE.mailBox as mail
import matplotlib.pyplot as plt
# import BigMomWTS as bm
import FactorModelingFunctions as bmModel
import FactorBaseFunctions as bmBase
# import scipy.optimize as sco
from datetime import datetime, date
import time
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
# import itertools
import logging

#### -1 logging settings
root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)

logging.basicConfig(filename=f'{root_path}log/log.txt',
                 format = '%(levelname)s - %(message)s',
                 level=logging.INFO)

today = date.today()

print(f'''当前交易日： {today}\n''')

logging.info(f'Run Time: {time.ctime()}\n')

logging.info(f'''1 - Current Trading day : {today}\n''')

#### 0 载入参数
with open('config/config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
    configFile = yaml.safe_load(parameters)

paramsBank = configFile['params']

paramWeightGenerator = configFile['weightGenerator']

pathBank = configFile['path']

pathFactor_weight = configFile['factor_weight']

locals().update(paramsBank['basic'])
locals().update(paramWeightGenerator)
locals().update(pathBank)
locals().update(pathFactor_weight)

del configFile, paramsBank, pathBank, pathFactor_weight,paramWeightGenerator

logging.info('2 - Load configurations\n')


## 时间表
cal_date = bmBase.get_rebalance_date(filepath_cal_date, dateNum)[start_date:]

cal_date = cal_date[cal_date.is_open == 1]

logging.info('3 - Load cal_date.csv\n')


if cal_date.loc[today.strftime('%Y-%m-%d'), 'rebalance_date'] or rebalance_trigger:
    print('4 - Rebalancing in process...\n')
    logging.info('4 - Rebalancing in process...\n')

    #% laod basic
    ### 1 基础信息表
    future_info, trade_cols, _, _ = bmBase.load_basic_info(filepath_future_list, filepath_factorTable2023,'trade')
    print('        4.1 - Load basic info\n')
    logging.info('        4.1 - Load basic info\n')
    
    ### 2 品种指数日数据
    price, rets, retMain, cost = bmBase.load_local_data(filepath_index,filepath_factorsF,future_info, trade_cols, start_date, today)
    
    print(f'        4.2 - Load updated datasets\n')
    logging.info(f'        4.2 - Load updated datasets\n')
    
    ### 3 因子池文件
    df_factorPools, periods = bmBase.load_factorPools(filepath_factorPools)
    
    print(f'        4.3 - Load factor pool\n')
    logging.info(f'        4.3 - Load factor pool\n')
    
    ### 4 forward return
    forward_returns = bmBase.compute_forward_returns(rets, periods)
    
    ### 5 生产因子
    logging.info(f'        4.4 - Generating factors - {time.ctime()}\n')
    
    try:
        df_f_rets, df_f_signals = bmModel.factorGenerator(df_factorPools, price, rets, forward_returns, cost,
                                                          trade_cols, filepath_factorPools, timeConsumption=2)
        print(f'        4.5 - Generate factors\n')
        
        logging.info(f'        4.5 -  Factors update process is completed - {time.ctime()}\n')
    
    except Exception as e:
        logging.error(e)
        
        df_f_rets = pd.read_csv('data/factorsDailyRet.csv', index_col=0, parse_dates=True)
        
        df_f_signals = pd.read_csv('data/factorsDailySignal.csv', index_col=[0,1], parse_dates=True)
        
        logging.info(f'        4.5 -  Use previous files\n')

    
    n_prod = df_f_rets.shape[1] # 实盘因子总数量
    
    logging.info(f'                Number of factors applied : {n_prod} \n')
    
    logging.info(f'                Max weight constraint: {maxW * 100}%\n')
    
    logging.info(f'                Lookback window size: {window_size}\n')
    
    close = price.loc[(slice(None), 'close'), :].droplevel(1)[trade_cols]
    
    df_rets_slice = df_f_rets.iloc[-window_size:, ]

    df_capital = pd.read_csv(filepath_capital, index_col='Name')
    
    list_files = []
    
    for objective in list_objective:
        
        logging.info(f'        4.6 - {objective} - Generating  position file\n')
        
        weight = pd.Series(eval(f'bmModel.{objective}(df_rets_slice, n_prod, maxW)'), index=df_f_rets.columns)

        signal = bmModel.signal_generator_real(weight, df_f_signals, rets.index, rets.columns)
        
        ### 3 交易文件输出        
        for acc in df_capital.index:
            
            capital = (df_capital.Cap * df_capital.Unit).loc[acc]
            
            print(f'account_{acc} total capital: {capital / 1000000} million\n')
            
            fileName = f'''account_{acc}/CTA_BigMom_{objective}_{acc}.csv'''
            
            outputCap = bmModel.output_position(signal, capital, future_info, close, fileName,
                                                filepath_templete, filepath_tradePosition)
            

            logging.info(f"""            4.6 - {objective} - {acc} - position files generated @ {fileName}\n""")
            
            logging.info(f"""            4.6 - {objective} - {acc} - total position = {round(outputCap.abs().sum(axis=1)[-1]/1000000,2)}unit\n""")
            
            logging.info(f"""            4.6 - {objective} - {acc} - net exposure = {round(outputCap.sum(axis=1)[-1]/1000000,2)}unit\n""")
        
            list_files.append(f'{filepath_tradePosition}{fileName}')
    
    
    receiverList = pd.read_excel('config/BMreceiveList.xlsx')
    
    receiverList = ','.join(receiverList['address'])
    
    mail.send_mime_mail('BigMomTradeFiles', receiverList, *[fileName for fileName in list_files])
    
    logging.info(f'5 - Sending trade files via email to {receiverList}\n')
    
else:
    
    rebalance_date = cal_date.loc[cal_date.rebalance_date, 'rebalance_date']
    
    rebalance_date_pre = rebalance_date[:today].index[-1].date()
    
    rebalance_date_next = rebalance_date[today:].index[0].date()
    
    logging.info(f'''4 - Previous Rebalance Date： {rebalance_date_pre}\n
                     Next Rebalance Date： {rebalance_date_next}\n''')

logging.info(f'''
                 E================================
                     N================================
                         D================================''')





    
