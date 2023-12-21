# -*- coding: utf-8 -*-
"""

因子时序分析

Created on Tue Nov 28 11:21:08 2023

Edited on Tue Nov 28 11:21:08 2023


@author: oOoOo_Andra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller


class timeseries_analysis():
    def __init__(self, df):
        
        self.df = df
        
        self.colName = df.columns[0]
        
        self.dfD = self.get_date()
        
    def get_date(self):
        dfD = self.df.copy()
        
        dfD['date'] = pd.to_datetime(dfD.index)
        
        dfD['year'] = [d.year for d in dfD.date]
        
        dfD['month'] = [d.strftime('%b') for d in dfD.date]
        
        return dfD
    
    #--趋势分析
    def trend_plot(self, window=100):
        #Determing rolling statistics
        mean = self.df.rolling(window).mean()
        std = self.df.rolling(window).std()
    
        #Plot rolling statistics:
        fig, ax = plt.subplots()
        ax.plot(self.df, color='blue',label=self.colName)
        ax.plot(mean, color='red', lw=2, label='Rolling Mean')
        ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(std, color='black', alpha=0.5, label = 'Rolling Std')
        ax.legend(loc='best')
        ax2.legend(loc=4)

    
    
    #-- 季节性分析
    def seasonal_analysis(self,period=252):
        decomposition = sm.tsa.seasonal_decompose(self.df, model='additive', period=period)
        
        decomposition.plot()
        
        return decomposition
    
    
    def seasonal_plot(self):
        
        pv = pd.pivot_table(self.df, index=self.df.index.month, columns=self.df.index.year,
                            values=self.colName, aggfunc='last')
        pv.plot()
        plt.title(f"Seasonal Plot of {self.colName} Time Series")
        plt.grid(True)
        
        
        
    def stationarity_test(self):
    
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(self.df, autolag='AIC')  #autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
        
        dfoutput = pd.Series(dftest[0:4], 
                             index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        
        print(dfoutput)
        
        return  dfoutput
    
    def ACF_plot(self):
        dfM = self.dfD.resample('M').last()[[self.colName]]
        
        dfM_S = dfM.diff().diff(12)
        
        fig, (ax1,ax2) = plt.subplots(2,1)
        sm.graphics.tsa.plot_acf(dfM_S.iloc[13:], ax=ax1)
        sm.graphics.tsa.plot_pacf(dfM_S.iloc[13:], ax=ax2)
        return fig
        
        
    def predict(self):
        dfM = self.dfD.resample('M').last()[[self.colName]]
        
        mod = sm.tsa.statespace.SARIMAX(dfM, trend='n', order=(0,1,0), seasonal_order=(0,1,1,12))
        
        results = mod.fit()
        
        print(results.summary())
        
        dfM['forecast'] = results.predict(start = dfM.shape[0] - 12, end= dfM.shape[0], dynamic= True)
        
        dfM.plot() 
        
        return dfM

if __name__ == '__main__':
    
    df = pd.DataFrame(np.random.randn(3000),columns=['value'])
    
    ts = timeseries_analysis(df)


    ts.trend_plot()


    ts.predict()

