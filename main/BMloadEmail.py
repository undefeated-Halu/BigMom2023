# -*- coding: utf-8 -*-
"""
从邮箱下载交易文件

Created on Wed Dec 20 16:21:22 2023

Edited on Wed Dec 20 16:21:22 2023


@author: oOoOo_Andra
"""

root_path = 'D:/ProgramFiles/python/strategy_factor/BigMom2023/'

root_data = 'D:/Data/'

import pandas as pd
import numpy as np
import os
os.chdir(root_path)
import sys
sys.path.append(f'{root_path}main')
sys.path.append('D:/ProgramFiles/python/')

import BASE.mailBox as mail
import time 
from datetime import datetime
import shutil

# 文件路径
filepath_capital = f'{root_path}config/BigMom2023_capital.csv'
filepath_tradePosition = f'{root_path}position_trade/'


for _ in range(10):
    
    res = mail.receive_mail(target='BigMomTradeFiles', download_dir=filepath_tradePosition)
    
    if len(res[1]) > 0:
        print(res[0])
        break
    else:
        print('retry')
        time.sleep(2)
    

df_capital = pd.read_csv(filepath_capital, index_col='Name')

move = True #是否转移相应文件

if move:
    file_paths= res[1]
    
    dates = [datetime.strptime(file_path.split('/')[-2], '%Y%b%d_%H%M') for file_path in file_paths]
    
    latest_date = max(dates)
    
    latest_file_paths = [file_path for file_path in file_paths if datetime.strptime(file_path.split('/')[-2], '%Y%b%d_%H%M') == latest_date]
    
    for file_path in latest_file_paths:
        for acc in df_capital.index:           
            if acc in file_path:
                version = df_capital.loc[acc, 'Version']
                if version in file_path:
                    shutil.copy(file_path, f'{filepath_tradePosition}account_{acc}')
                    print(f'{filepath_tradePosition}account_{acc}')
            
        
    


