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

root_path = 'D:/ProgramFiles/python/strategy_factor/BigMom2023/'

root_data = 'D:/Data/'

import pandas as pd
import numpy as np
# import os
# os.chdir(root_path)
import yaml
import sys
sys.path.append(f'{root_path}main')
# import ctaBasicFunc as cta
# from datetime import datetime
import matplotlib.pyplot as plt
import BigMomWTS as bm
import FactorModelingFunctions as bmModel
import FactorBaseFunctions as bmBase
import scipy.optimize as sco
from datetime import datetime
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

#% laod basic
### 1 基础信息表
test_date = 'factorCompositionDec1'

Description = ''

future_info, trade_cols, list_factor_test,df_factorTable = bm.load_basic_info(filepath_future_list, filepath_factorTable2023)

### 2-5 品种指数日数据
price, rets, retMain, cost = bm.load_local_data(filepath_index,filepath_factorsF,future_info, trade_cols, start_date, end_date)
# rets = retIndex[:end_date]


#--- 因子池文件已经
df_factorPools = pd.read_excel(filepath_factorPools, index_col=0)

# df_factorPools = df_factorPools[df_factorPools.tag_test]

df_factorPools['testCode'] = range(len(df_factorPools))

df_factorPools['hp'] = df_factorPools.param.apply(lambda x: eval(x)[-1])

periods = df_factorPools['hp'].unique()

print(f'''factorPools长度： {df_factorPools.shape[0]}''')

#--- forward return
forward_returns = bmBase.compute_forward_returns(rets, periods)

forward_returns_group = bmBase.factor_fast_groupon(forward_returns, groupNum=2)

#%% 生成 数据

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

#%% trim dat

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
df_factorPools.to_excel(filepath_factorPools)


#%% 因子组合 1
dt = '2023'

n_prod = df_f_rets.shape[1]

ret_ptf = pd.DataFrame()

#--- 0 因子等权
 
weight = pd.Series(1/n_prod, index=df_f_rets.columns)

signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)

for i in range(len(weight)):
    
    signal += df_f_signals.xs(i, level=1) * weight[i]

signal = signal.div(signal.abs().sum(axis=1), axis=0)

ret_ptf['avg'] = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()


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
# ax.plot(ret_ptf['average'][:date_out], 'g', label='average')
# ax.plot(ret_ptf['icir'][:date_out], 'b', label='sr')
ax.plot(ret_, 'k', label='sco_fee')
ax.plot(ret_[dt:], 'r', lw=2, label='outSample')
ax.legend(loc=2)
ax.grid(True)

ret_ptf['outSample'] = ret_
#%% 因子组合 2
#--- 推进
import FactorModelingFunctions as bmModel

dfweight = pd.DataFrame(np.NaN, index=rets.index, columns=df_f_rets.columns)

# 滚动回测
window_size = 250

step_size = 20

n_day = dfweight.shape[0]

list_func = ['max_sp','max_sortino','min_volatility','max_upside_volatility','min_downside_volatility']

for func in list_func:
    print(func)

    for i in range(window_size, n_day, step_size):
        print(df_f_rets.index[i])
        
        df_rets_slice = df_f_rets.iloc[i-window_size:i,]
        
        opts = sco.minimize(
                            fun = eval(f'bmModel.{func}'),
                            # fun = bmModel.min_sp,
                            # fun = bmModel.min_sortino,
                            x0  = n_prod * [1./n_prod,] ,
                            method = 'SLSQP' ,
                            args = (df_rets_slice ) ,
                            bounds = bnds ,
                            constraints = cons)
        
        res = opts.x
            
        dfweight.iloc[i,:] = res


    # 填充nan值
    dfweightCopy = dfweight.fillna(method='ffill').fillna(1/dfweight.shape[1])
    
    signal = pd.DataFrame(0, index=rets.index, columns=rets.columns)
    
    for i in range(len(res)):
        
        signal += df_f_signals.xs(i, level=1).mul(dfweightCopy.iloc[:,i], axis=0).fillna(0)
    
    signal = signal.div(signal.abs().sum(axis=1), axis=0)
    

    ret_ptf[func] = (signal.shift() * rets -  signal.diff() * cost).sum(axis=1).cumsum()



perf_ratios = pd.DataFrame(columns=ret_ptf.columns,index=['ar','sr','mdd','mar'])

_, ax = plt.subplots()

# ax.plot(ret_ptf, 'k', label=f'back{window_size}forward{step_size}')
for col in ['avg', 'outSample'] + list_func:
    perf_ratios[col] = bmBase.performance_ratio(ret_ptf[col].diff()[window_size:].fillna(0))
    
    ax.plot(ret_ptf[col][window_size:].diff().cumsum(), label=col, alpha=0.5)
    
ax.plot(ret_ptf['outSample'][window_size:].diff().cumsum()[dt:], 'r', lw=2)
# ax.plot(ret_ptf['outSample'][dt:], 'r', lw=2, label='outSample')

ax.set_title(f'back{window_size}forward{step_size}')
ax.legend(loc=2)

ax.grid(True)


print(perf_ratios )

perf_ratios.to_csv('output/perf_ratios.csv')


ret_ptf[['max_sp','min_volatility','min_downside_volatility']][window_size:].plot()
#%% ml data preparation

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA

data = pd.concat([df_f, forward_returns])

data.reset_index('factor', drop=False, inplace=True)


training_sample = data.loc[data.index < '2023-3-31'].set_index('factor',append=True).stack().unstack(1)
testing_sample = data.loc[data.index > '2023-3-31'].set_index('factor',append=True).stack().unstack(1)


y_train = training_sample['1D'].fillna(0).values
X_train = training_sample[list_features].fillna(0).values


y_test = testing_sample['1D'].fillna(0).values
X_test = testing_sample[list_features].fillna(0).values

ret = testing_sample.loc[:,['1D']].unstack(1)
ret.columns = ret.columns.droplevel(0)


pca = PCA(n_components = 10)

X = pca.transform(data[list_features])



#%% PCA 


#%%Tree-based methods

def predict_return(predict, testing_sample,ret, method='regression'):
    testing_sample['predict'] = predict
    
    dfpos = testing_sample.loc[:,['predict']].unstack(1)
    
    dfpos.columns = dfpos.columns.droplevel(0)
    
    if method=='regression':
        dfpos = 1 * (dfpos>0) - 1 * (dfpos <0)
    if method == 'classifaction':
        dfpos = 1 * (dfpos < 2) - 1 * (dfpos >4)
    
    dfpos = dfpos.div(dfpos.abs().sum(axis=1), axis=0)
        
    (dfpos * ret).sum(axis=1).cumsum().plot()

#--- Lasso
# 使用Lasso回归选择因子
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
selected_factors = factors[:, lasso.coef_ != 0]

# 使用线性回归加权因子
lr = LinearRegression()
lr.fit(selected_factors, returns)
weights = lr.coef_ / lr.coef_.sum()

# 使用权重构建投资组合
portfolio = (selected_factors * weights).sum(axis=1)

# 计算投资组合的收益和风险
portfolio_return = (returns * weights).sum()
portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(selected_factors.T), weights)))

# 输出结果
print("Selected factors:\n", selected_factors)
print("Weights:\n", weights)
print("Portfolio return:", portfolio_return)
print("Portfolio std:", portfolio_std)



#--- LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)    


predict = lin_reg.predict(X_test)
mse = np.mean(( predict- y_test)**2)
hitratio = np.mean(predict * y_test > 0)
print(f'DecisionTreeRegressor\nMSE: {mse} \nHit Ratio: {hitratio}')

predict_return(predict, testing_sample)

# Trees
# from xgboost import XGBClassifier

# =============================================================================
# y = forward_returns.xs('10D', level='hp').unstack().fillna(0).values
# X = df_f.stack().unstack('factor').fillna(0).values
# 
# clf = tree.DecisionTreeRegressor(max_depth=3, ccp_alpha=1e-4)
# clf = clf.fit(X,y)
# 
# dot_data = tree.export_graphviz(clf,out_file=None,
#                                 feature_names=features,
#                                 filled = True)
# 
# graph = graphviz.Source(dot_data)
# graph
# 
# 
# 
# cols_that_matter = []
# 
# for val in np.where(clf.feature_importances_ > 0)[0]:
#   cols_that_matter.append(features[val])
# 
# cols_that_matter
# 
# temp = pd.melt(data[['R1M_Usd']+cols_that_matter], id_vars='R1M_Usd')
# 
# sns.lineplot(data = temp, y='R1M_Usd', x='value', hue='variable');
# 
# =============================================================================


#--- 决策树回归

dt = tree.DecisionTreeRegressor(max_depth=3, ccp_alpha=1e-4)
dt.fit(X_train,y_train)


mse = np.mean((dt.predict(X_test) - y_test)**2)
hitratio = np.mean(dt.predict(X_test) * y_test > 0)
print(f'DecisionTreeRegressor\nMSE: {mse} \nHit Ratio: {hitratio}')

predict_return(dt.predict(X_test), testing_sample)



#--- 随机森林回归
rf = RandomForestRegressor(n_estimators=40,
                           max_features = 30,
                           min_samples_split=10000,
                           bootstrap=False)
rf.fit(X_train,y_train)

mse = np.mean((rf.predict(X_test) - y_test)**2)
hitratio = np.mean(rf.predict(X_test) * y_test > 0)
print(f'RandomForestRegressor\nMSE: {mse} \nHit Ratio: {hitratio}')


predict_return(rf.predict(X_test), testing_sample)


data = pd.concat([df_f, forward_returns_group])

data.reset_index('factor', drop=False, inplace=True)

training_sample = data.loc[data.index < '2023-3-31'].set_index('factor',append=True).stack().unstack(1)
testing_sample = data.loc[data.index > '2023-3-31'].set_index('factor',append=True).stack().unstack(1)


y_train = training_sample['5D'].fillna(0).values
X_train = training_sample[list_features].fillna(0).values


y_test = testing_sample['5D'].fillna(0).values
X_test = testing_sample[list_features].fillna(0).values


#--- 随机森林分类
# y_train_clf = np.where(y_train>0,1,-1)

clf = RandomForestClassifier(n_estimators=40,
                              max_features = 30,
                              min_samples_split=10000,
                              bootstrap=False)
clf.fit(X_train,y_train)

# y_test_clf =  np.where(y_test>0,1,-1)

predict = clf.predict(X_test)
hitratio = np.mean(predict == y_test)
print(f'RandomForestClassifier\nHit Ratio: {hitratio}')

predict_return(predict, testing_sample, ret)


#--- AdaBoost 分类

ada = AdaBoostClassifier()

ada.fit(X_train, y_train)

hitratio = np.mean(ada.predict(X_test) == y_test)
print(f'Hit Ratio: {hitratio}')

predict_return(ada.predict(X_test), testing_sample, ret)

# =============================================================================
# #--- XGBM 分类
# xgb = XGBClassifier()
# 
# xgb.fit(X_train, y_train_clf)
# 
# hitratio = np.mean(xgb.predict(X_test) == y_test_clf)
# print(f'Hit Ratio: {hitratio}')
# 
# =============================================================================


#%%






