# -*- coding: utf-8 -*-
"""
strategyName:
    BigMom2023 Factor DR
    
    dimension reduction

edition: 
    Playground
    
strategyType:
    strategy_factor
Description: 

TODOs:    
    PAC
    SVD
    
Created on Mon Dec  4 08:57:36 2023

Edited on Mon Dec  4 08:57:36 2023


@author: oOoOo_Andra
"""
##
# Python ≥3.5 is required

root_path = 'D:/ProgramFiles/python/strategy_factor/BigMom2023/'

root_data = 'D:/Data/'

# import os
# os.chdir(root_path)
import sys
sys.path.append(f'{root_path}main')

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import yaml

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
import FactorBaseFunctions as bmBase
import BigMomWTS as bm

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "dim_reduction"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    #print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

with open('config/config_BigMom2023.yml', 'r', encoding='utf-8') as parameters:
    configFile = yaml.safe_load(parameters)

paramsBank = configFile['params']

pathBank = configFile['path']

pathFactor_weight = configFile['factor_weight']

locals().update(paramsBank['basic'])
locals().update(pathBank)
locals().update(pathFactor_weight)

del configFile, paramsBank, pathBank, pathFactor_weight

#%%
df_f_r = pd.read_csv('data/factorsDailyRet.csv', index_col=0, parse_dates=True)

df_f = pd.read_csv('data/factors.csv', index_col=[0,1], parse_dates=True)

df_f_T = df_f.T.stack(level=0)

#%%
pca = PCA(n_components = 1)

data = df_f_T[df_f_T.isna().sum(axis=1) < 50]

reduced = pca.fit_transform(data.fillna(0).values)

#%
price, rets, retMain, cost = bm.load_local_data(filepath_index,filepath_factorsF,future_info, trade_cols, start_date, end_date)

df_reduced = pd.DataFrame(reduced, index=data.index)

df_dr = df_reduced.T.stack(level=1)

df_dr_rank = df_dr.rank(axis=1, ascending=False)

factor_signal = (1 * bmBase.grouper(df_dr_rank, 5, n=5, method='ceil') \
                 - 1 * bmBase.grouper(df_dr_rank, 5, n=(5 + 1 - 5), method='ceil'))

factor_pos = factor_signal.div(factor_signal.abs().sum(axis=1),axis=0).groupby(level=1).mean()

factor_pos = factor_pos.div(factor_pos.abs().sum(axis=1),axis=0)


dfpos = pd.concat([rets.iloc[:,0], factor_pos],axis=1).iloc[:,1:]

dfret = dfpos.shift() * rets

dfcumret = dfret.sum(axis=1).cumsum()

dfcumret.plot()
