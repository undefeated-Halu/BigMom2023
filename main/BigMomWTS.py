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
sys.path.append(f'{path_strategy}main/')

import bottleneck as bn
from FactorCalcFunctions import *
from FactorBaseFunctions import *
import yaml
import ctaBasicFunc as cta
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class WTS_factor:
    def __init__(self, price, symbol_list):
        # benchmark_price = get_price(index, None, end_date, '1d',['open','close','low','high','avg_price','prev_close','volume'], False, None, 250,is_panel=1)
        self.OPEN = price.loc[(slice(None), 'open'), :].droplevel(1)[symbol_list]
        self.CLOSE = price.loc[(slice(None), 'close'), :].droplevel(1)[symbol_list]
        self.LOW = price.loc[(slice(None), 'low'), :].droplevel(1)[symbol_list]
        self.HIGH = price.loc[(slice(None), 'high'), :].droplevel(1)[symbol_list]
        # volume weighted average price
        self.VWAP = price.loc[(slice(None), 'avg'), :].droplevel(1)[symbol_list]
        # market value
        self.MKTV = price.loc[(slice(None), 'mkt_value'), :].droplevel(1)[symbol_list]
        # trade fees
        self.COST = price.loc[(slice(None), 'cost'), :].droplevel(1)[symbol_list]
        # market_weight
        self.MKTW = self.MKTV.div(self.MKTV.sum(axis=1), axis=0)
        self.PRE_CLOSE = price.loc[(slice(None), 'pre_close'), :].droplevel(1)[symbol_list]
        self.VOLUME = price.loc[(slice(None), 'volume'), :].droplevel(1)[symbol_list]
        self.AMOUNT = price.loc[(slice(None), 'amount'), :].droplevel(1)[symbol_list]
        self.RET = price.loc[(slice(None), 'close'), :].droplevel(1).pct_change().fillna(0)[symbol_list]
        
        self.OI = price.loc[(slice(None), 'oi'), :].droplevel(1)[symbol_list]
        # self.HD         = self.HIGH - DELAY(self.HIGH,1)
        # self.LD         = DELAY(self.LOW,1) - self.LOW 
        # self.TR         = MAX(MAX(self.HIGH-self.LOW, ABS(self.HIGH-DELAY(self.CLOSE,1))),ABS(self.LOW-DELAY(self.CLOSE,1)))


    #--- 1-50
    def alpha_001(self, N=6):
        # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
        return (-1 * CORR(RANK(DELTA(LOG(self.VOLUME), 1)), RANK(((self.CLOSE - self.OPEN) / self.OPEN)), N))

    def alpha_002(self, N):
        # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
        return -1 * DELTA(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW), N)

    def alpha_003(self, N=6):
        # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
        return SUM(IFELSE(self.CLOSE == DELAY(self.CLOSE, 1), 0,
                          self.CLOSE - IFELSE(self.CLOSE > DELAY(self.CLOSE, 1),
                                              MIN(self.LOW, DELAY(self.CLOSE, 1)),
                                              MAX(self.HIGH, DELAY(self.CLOSE, 1)))), N)

    def alpha_004(self, N=8, M=2):
        '''
        ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) :
            (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 :
                (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        '''
        return IFELSE((((SUM(self.CLOSE, N) / N) + STD(self.CLOSE, N)) < (SUM(self.CLOSE, M) / M)), -1,
                      IFELSE(((SUM(self.CLOSE, M) / M) < ((SUM(self.CLOSE, N) / N) - STD(self.CLOSE, N))), 1,
                             IFELSE(((1 < (self.VOLUME / MEAN(self.VOLUME, 20))) | (
                                         (self.VOLUME / MEAN(self.VOLUME, 20)) == 1)), 1, -1)))

    def alpha_006(self, p=0.15):
        # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
        return RANK(SIGN(DELTA((self.OPEN * (1 - p)) + (self.HIGH * p), 4))) * -1

    def alpha_007(self, n=3):
        # ((RANK(MAX((avg_price - close), n)) + RANK(MIN((avg_price - close), n))) * RANK(DELTA(volume, n)))
        return ((RANK(MAX(self.VWAP - self.CLOSE, n))
                 + RANK(MIN((self.VWAP - self.CLOSE), n))) * RANK(DELTA(self.VOLUME, n)))

    def alpha_008(self, P=0.2):
        # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
        return RANK(DELTA(((((self.HIGH + self.LOW) / 2) * P) + (self.VWAP * (1 - P))), 4) * -1)

    def alpha_009(self, n=7):
        # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
        # LVP -SMA
        return -SMA(((self.HIGH + self.LOW) / 2 - (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1)) / 2)
                    * (self.HIGH - self.LOW) / self.VOLUME, n, 2)

    def alpha_010(self, N=20):
        # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
        # LVP -
        return - RANK(MAX(IFELSE(self.RET < 0, STD(self.RET, N), self.CLOSE) ** 2, 5))

    def alpha_011(self, N=6):
        # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
        return SUM(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW) * self.VOLUME, N)

    def alpha_012(self, N=10):
        # (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
        return (RANK((self.OPEN - (SUM(self.VWAP, N) / N)))) * (-1 * (RANK(ABS((self.CLOSE - self.VWAP)))))

    def alpha_013(self, N):
        # (((HIGH * LOW)**0.5) - VWAP)
        return -(((self.HIGH * self.LOW) ** N) - self.VWAP)  # -1

    def alpha_014(self, N=5):
        # CLOSE-DELAY(CLOSE,5)
        return -(self.CLOSE - DELAY(self.CLOSE, N))  # -1

    def alpha_015(self, N=1):
        # OPEN/DELAY(CLOSE,1)-1 LVP: OPEN/DELAY(CLOSE,N)-1
        return -self.OPEN / DELAY(self.CLOSE, N) - 1

    def alpha_016(self, N=5):
        # (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        return (-1 * TSMAX(RANK(CORR(RANK(self.VOLUME), RANK(self.VWAP), N)), N))

    def alpha_017(self, N=15):
        # RANK((VWAP - MAX(VWAP, 15)))**DELTA(CLOSE, 5)
        return RANK((self.VWAP - MAX(self.VWAP, N))) ** DELTA(self.CLOSE, 5)

    def alpha_018(self, N=5):
        ## CLOSE/DELAY(CLOSE,5)
        return self.CLOSE / DELAY(self.CLOSE, N)

    def alpha_019(self, N=5):
        # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
        return (IFELSE(self.CLOSE < DELAY(self.CLOSE, N),
                       (self.CLOSE - DELAY(self.CLOSE, N)) / DELAY(self.CLOSE, N),
                       IFELSE(self.CLOSE == DELAY(self.CLOSE, N), 0, (self.CLOSE - DELAY(self.CLOSE, N)) / self.CLOSE)))

    def alpha_020(self, N=6):
        # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        return (self.CLOSE - DELAY(self.CLOSE, N)) / DELAY(self.CLOSE, N) * 100

    def alpha_021(self, N=6):
        # REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
        return REGBETA(MEAN(self.CLOSE, N), SEQUENCE(N))

    def alpha_022(self, N=12):
        # MEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
        M = int(N / 2)
        P = int(N / 4)
        temp1 = (self.CLOSE - MEAN(self.CLOSE, M)) / MEAN(self.CLOSE, M)
        temp2 = DELAY(temp1, P)
        return SMA(temp1 - temp2, N, 1)

    def alpha_023(self, N=20):
        # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+ SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
        temp1 = IFELSE(self.RET > 0, STD(self.CLOSE, N), 0)
        temp2 = SMA(temp1, N, 1)
        temp3 = SMA(IFELSE(self.RET <= 0, STD(self.CLOSE, N), 0), N, 1)

        return temp2 / (temp2 + temp3) * 100

    def alpha_024(self, N=5):
        # SMA(CLOSE-DELAY(CLOSE,5),5,1)
        return SMA((self.CLOSE / DELAY(self.CLOSE, N) - 1), N, 1)

    def alpha_025(self, N=20, M=7, P=9):
        # ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 +RANK(SUM(VWAP, 250))))
        return ((-1 * RANK((DELTA(self.CLOSE, M)
                            * (1 - RANK(DECAYLINEAR((self.VOLUME / MEAN(self.VOLUME, N)), P)))))) *
                (1 + RANK(SUM(self.VWAP, 250))))

    def alpha_026(self, N=7, M=5):
        # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
        return ((((SUM(self.CLOSE, N) / N) / self.CLOSE) - 1) + ((CORR(self.VWAP, DELAY(self.CLOSE, M), 230))))

    def alpha_027(self, N=3):
        # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
        temp1 = DELAY(self.CLOSE, N)
        temp2 = DELAY(self.CLOSE, 2 * N)
        return WMA((self.CLOSE - temp1) / temp1 * 100
                   + (self.CLOSE - temp2) / temp2 * 100, 4 * N)

    def alpha_028(self, N=9, M=3):
        # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
        temp1 = TSMIN(self.LOW, N)
        return (3 * SMA((self.CLOSE - temp1) / (TSMAX(self.HIGH, N) - temp1) * 100, M, 1)
                - 2 * SMA(SMA((self.CLOSE - temp1) / (MAX(self.HIGH, N) - TSMAX(self.LOW, N)) * 100, M, 1), M, 1))

    def alpha_029(self, N=6):
        # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        return (self.CLOSE - DELAY(self.CLOSE, N)) / DELAY(self.CLOSE, N) * self.VOLUME

    def alpha_031(self, N=12):
        # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
        temp = -MEAN(self.CLOSE, N)
        return (self.CLOSE - temp) / temp * 100

    def alpha_032(self, N=3):
        # (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        return (-1 * SUM(RANK(CORR(RANK(self.HIGH), RANK(self.VOLUME), N)), N))

    def alpha_033(self, N=5):
        # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))
        temp = TSMIN(self.LOW, N)
        return ((((-1 * temp) + DELAY(temp, N))
                 * RANK(((SUM(self.VWAP, 240) - SUM(self.VWAP, N * 4)) / 220)))
                * TSRANK(self.VOLUME, N))

    def alpha_034(self, N=12):
        # MEAN(CLOSE,12)/CLOSE
        return MEAN(self.CLOSE, N) / self.CLOSE

    def alpha_035(self, N=15, M=7, P=0.65):
        # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)
        return (MIN(RANK(DECAYLINEAR(DELTA(self.OPEN, 1), N)),
                    RANK(DECAYLINEAR(CORR((self.VOLUME), ((self.OPEN * P) + (self.OPEN * (1 - P))), 17), M))) * -1)

    def alpha_036(self, N=6):
        # RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
        return RANK(SUM(CORR(RANK(self.VOLUME), RANK(self.VWAP), N), 2))

    def alpha_037(self, N=5):
        # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
        return (-1 * RANK(((SUM(self.OPEN, N) * SUM(self.VWAP, N))
                           - DELAY((SUM(self.OPEN, N) * SUM(self.VWAP, N)), N * 2))))

    def alpha_038(self, N=20):
        # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        return IFELSE((SUM(self.HIGH, N) / N < self.HIGH), (-1 * DELTA(self.HIGH, 2)), 0)

    def alpha_039(self, N=8, M=12, P=0.3):
        # ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
        return ((RANK(DECAYLINEAR(DELTA((self.CLOSE), 2), N))
                 - RANK(DECAYLINEAR(CORR(((self.VWAP * P) + (self.OPEN * (1 - P))),
                                         SUM(MEAN(self.VOLUME, 180), 37), 14), M))) * -1)

    def alpha_040(self, N=26):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
        return (SUM(IFELSE(self.RET > 0, self.VOLUME, 0), N)
                / SUM(IFELSE(self.RET <= 0, self.VOLUME, 0), N) * 100)

    def alpha_041(self, N=5, M=3):
        # (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
        return (RANK(MAX(DELTA((self.VWAP), M), N)) * -1)

    def alpha_042(self, N=10):
        # ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
        return ((-1 * RANK(STD(self.HIGH, N))) * CORR(self.HIGH, self.VOLUME, N))

    def alpha_043(self, N=6):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
        # SUM(IFELSE(RET>0, VOLUME, IFELSE(RET<=0,-VOLUME,0)),6)
        return SUM(IFELSE(self.RET > 0, self.VOLUME, IFELSE(self.RET <= 0, -self.VOLUME, 0)), N)

    def alpha_044(self, N=6, M=10):
        # (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))
        return (TSRANK(DECAYLINEAR(CORR(((self.LOW)), MEAN(self.VOLUME, 10), 7), N), 4)
                + TSRANK(DECAYLINEAR(DELTA((self.VWAP), 3), M), 15))

    def alpha_045(self, N=15, P=0.6):
        # (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
        return (RANK(DELTA((((self.CLOSE * P) + (self.OPEN * (1 - P)))), 1))
                * RANK(CORR(self.VWAP, MEAN(self.VOLUME, 150), N)))

    def alpha_046(self, N=3):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
        return (MEAN(self.CLOSE, N) + MEAN(self.CLOSE, 2 * N) + MEAN(self.CLOSE, 4 * N) + MEAN(self.CLOSE, 6 * N)) / (
                    4 * self.CLOSE)

    def alpha_047(self, N=6):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
        return SMA((TSMAX(self.HIGH, N) - self.CLOSE) / (TSMAX(self.HIGH, N) - TSMIN(self.LOW, N)) * 100, 9, 1)

    def alpha_048(self, N=5):
        # (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3))))))
        # * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
        return (-1 * ((RANK(((SIGN((self.CLOSE - DELAY(self.CLOSE, 1)))
                              + SIGN((DELAY(self.CLOSE, 1) - DELAY(self.CLOSE, 2))))
                             + SIGN((DELAY(self.CLOSE, 2) - DELAY(self.CLOSE, 3))))))
                      * SUM(self.VOLUME, N)) / SUM(self.VOLUME, 4 * N))

    def alpha_049(self, N=12):
        # (SUM(IFELSE((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        #   /(SUM(IFELSE((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        #     +SUM(IFELSE((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)))
        temp1 = self.HIGH + self.LOW
        temp2 = DELAY(self.HIGH, 1) + DELAY(self.LOW, 1)
        temp3 = MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))
        temp4 = SUM(IFELSE(temp1 > temp2, 0, temp3), N)

        return (temp4 / (temp4 + SUM(IFELSE(temp1 < temp2, 0, temp3), N)))
    #--- 51-100
    def alpha_051(self, N=12):
        # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        # /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        #   +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        left = self.HIGH + self.LOW
        right = DELAY(self.HIGH, 1) + DELAY(self.LOW, 1)
        temp1 = left <= right
        temp2 = left >= right
        res = MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))
        temp3 = SUM(IFELSE(temp1, 0, res), N)
        temp4 = SUM(IFELSE(temp2, 0, res), N)
        return temp3 / (temp3 + temp4)

    def alpha_052(self, N=26):
        # SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
        temp = DELAY((self.HIGH + self.LOW + self.CLOSE) / 3, 1)
        return (SUM(MAX(0, self.HIGH - temp), N) / SUM(MAX(0, temp - self.LOW), N))

    def alpha_053(self, N=12):
        # COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
        return COUNT(self.CLOSE > DELAY(self.CLOSE, 1), N) / N * 100

    def alpha_054(self, N=10):
        # (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
        return (-1 * RANK(
            (STD(ABS(self.CLOSE - self.OPEN)) + (self.CLOSE - self.OPEN)) + CORR(self.CLOSE, self.OPEN, N)))

    def alpha_055(self, N=20):
        # SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))
        #     /((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))
        #        ?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
        #        :(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))
        #          ?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
        #          :ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
        #     *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
        temp1 = 16 * (self.CLOSE + (self.CLOSE - self.OPEN) / 2 - DELAY(self.OPEN, 1))
        temp2 = ABS(self.HIGH - DELAY(self.CLOSE, 1))
        temp3 = ABS(self.LOW - DELAY(self.CLOSE, 1))
        temp4 = ABS(self.HIGH - DELAY(self.LOW, 1))
        temp5 = MAX(temp2, temp3)
        return SUM(temp1 / (IFELSE(((temp2 > temp3) & (temp2 > temp4)),
                                   (temp2 + temp3) / 2 + temp5,
                                   IFELSE(((temp3 > temp4) & (temp3 > temp2)),
                                          (temp3 + temp2) / 2 + temp5,
                                          temp4 + temp5))) * temp5, N)

    def alpha_056(self, N=19, M=13):
        # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))**5)))
        return 1 * (RANK((self.OPEN - TSMIN(self.OPEN, 12))) < RANK((RANK(CORR(SUM(((self.HIGH + self.LOW) / 2), N),
                                                                               SUM(MEAN(self.VOLUME, 40), N),
                                                                               13)) ** 5)))

    def alpha_057(self, N=9):
        return SMA((self.CLOSE - TSMIN(self.LOW, N)) / (TSMAX(self.HIGH, N) - TSMIN(self.LOW, N)) * 100, 3, 1)

    def alpha_058(self, N=20):
        # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        return COUNT(self.CLOSE > DELAY(self.CLOSE, 1), N) / N * 100

    def alpha_059(self, N=20):
        # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
        temp = DELAY(self.CLOSE, 1)
        return (SUM(IFELSE(self.CLOSE == temp, 0, self.CLOSE - IFELSE(self.CLOSE > temp,
                                                                      MIN(self.LOW, temp),
                                                                      MAX(self.HIGH, temp))), N))

    def alpha_060(self, N=20):
        # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
        return SUM(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW) * self.VOLUME, N)

    def alpha_061(self, N=12, M=17):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
        return (MAX(RANK(DECAYLINEAR(DELTA(self.VWAP, 1), N)),
                    RANK(DECAYLINEAR(RANK(CORR((self.LOW), MEAN(self.VOLUME, 80), 8)), M))) * -1)

    def alpha_062(self, N=5):
        # (-1 * CORR(HIGH, RANK(VOLUME), 5))
        return (-1 * CORR(self.HIGH, RANK(self.VOLUME), N))

    def alpha_063(self, N=6):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
        return SMA(MAX(self.CLOSE - DELAY(self.CLOSE, 1), 0), N, 1) / SMA(ABS(self.CLOSE - DELAY(self.CLOSE, 1)), N,
                                                                          1) * 100

    def alpha_064(self,N):
        # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
        #      RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
        return (MAX(RANK(DECAYLINEAR(CORR(RANK(self.VWAP), RANK(self.VOLUME), 4), 4)),
                    RANK(DECAYLINEAR(MAX(CORR(RANK(self.CLOSE), RANK(MEAN(self.VOLUME, 60)), 4), 13), 14))) * -1)

    def alpha_065(self, N=6):
        # MEAN(CLOSE,6)/CLOSE
        return MEAN(self.CLOSE, N) / self.CLOSE

    def alpha_066(self, N=6):
        # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
        return (self.CLOSE - MEAN(self.CLOSE, N)) / MEAN(self.CLOSE, N) * 100

    def alpha_067(self, N=24):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
        temp = self.CLOSE - DELAY(self.CLOSE, 1)
        return SMA(MAX(temp, 0), N, 1) / SMA(ABS(temp), N, 1) * 100

    def alpha_068(self, N=15):
        # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
        return SMA(((self.HIGH + self.LOW) / 2 - (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1)) / 2)
                   * (self.HIGH - self.LOW) / self.VOLUME, N, 2)

    def alpha_070(self, N=6):
        # STD(AMOUNT, 6)
        return STD(self.AMOUNT, N)

    def alpha_071(self, N=24):
        # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        return (self.CLOSE - MEAN(self.CLOSE, N)) / MEAN(self.CLOSE, N) * 100

    def alpha_072(self, N=6, M=15):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        return SMA((TSMAX(self.HIGH, N) - self.CLOSE) / (TSMAX(self.HIGH, N) - TSMIN(self.LOW, 6)) * 100, M, 1)

    def alpha_073(self, N):
        # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5)
        # -RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
        return ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((self.CLOSE), self.VOLUME, 10), 16), 4), 5)
                 - RANK(DECAYLINEAR(CORR(self.VWAP, MEAN(self.VOLUME, 30), 4), 3))) * -1)

    def alpha_074(self, N=20, M=40, P=0.35):
        # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7))
        #  + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        return (RANK(CORR(SUM(((self.LOW * P) + (self.VWAP * (1 - P))), N), SUM(MEAN(self.VOLUME, M), N), 7))
                + RANK(CORR(RANK(self.VWAP), RANK(self.VOLUME), 6)))

    def alpha_076(self, N=20):
        # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        return (STD(ABS((self.CLOSE / DELAY(self.CLOSE, 1) - 1)) / self.VOLUME, N)
                / MEAN(ABS((self.CLOSE / DELAY(self.CLOSE, 1) - 1)) / self.VOLUME, N))

    def alpha_077(self, N=40):
        # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)),
        # RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
        return MIN(RANK(DECAYLINEAR(((((self.HIGH + self.LOW) / 2) + self.HIGH)
                                     - (self.VWAP + self.HIGH)), 20)),
                   RANK(DECAYLINEAR(CORR(((self.HIGH + self.LOW) / 2),
                                         MEAN(self.VOLUME, N), 3), 6)))

    def alpha_078(self, N=12, P=0.015):
        # ((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        temp = (self.HIGH + self.LOW + self.CLOSE) / 3
        return ((temp - MEAN(temp, N)) / (P * MEAN(ABS(self.CLOSE - MEAN(temp, N)), N)))

    def alpha_079(self, N=12):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        return SMA(MAX(self.CLOSE - DELAY(self.CLOSE, 1), 0), N, 1) / SMA(ABS(self.CLOSE - DELAY(self.CLOSE, 1)), N,
                                                                          1) * 100

    def alpha_080(self, N=5):
        # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        return (self.VOLUME - DELAY(self.VOLUME, N)) / DELAY(self.VOLUME, N) * 100

    def alpha_081(self, n=21):
        # SMA(VOLUME,21,2)
        return SMA(self.VOLUME, n, 2)

    def alpha_082(self, N=6, M=20):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
        return SMA((TSMAX(self.HIGH, N) - self.CLOSE) / (TSMAX(self.HIGH, N) - TSMIN(self.LOW, N)) * 100, M, 1)

    def alpha_083(self, N=5):
        # (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
        return (-1 * RANK(COVIANCE(RANK(self.HIGH), RANK(self.VOLUME), N)))

    def alpha_084(self, N=20):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
        return SUM(IFELSE(self.CLOSE > DELAY(self.CLOSE, 1),
                          self.VOLUME,
                          IFELSE(self.CLOSE < DELAY(self.CLOSE, 1), -self.VOLUME, 0)), N)

    def alpha_085(self, N=20, M=8):
        # (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
        return (TSRANK((self.VOLUME / MEAN(self.VOLUME, N)), N) * TSRANK((-1 * DELTA(self.CLOSE, 7)), M))

    def alpha_086(self, N=10):
        # ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :
        # (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) *
        # (CLOSE - DELAY(CLOSE, 1)))))
        temp1 = (DELAY(self.CLOSE, 2 * N) - DELAY(self.CLOSE, 10)) / 10
        temp2 = (DELAY(self.CLOSE, N) - self.CLOSE) / 10
        return IFELSE(0.25 < (temp1 - temp2),
                      -1,
                      IFELSE((temp1 - temp2) < 0,
                             1,
                             -1 * (self.CLOSE - DELAY(self.CLOSE, 1))))

    #############################################################################
    def alpha_087(self, N=7, M=11, P=0.9):
        # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7))
        #   + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP)
        #                         /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
        return ((RANK(DECAYLINEAR(DELTA(self.VWAP, 4), N))
                 + TSRANK(DECAYLINEAR(((((self.LOW * P) + (self.LOW * (1 - P))) - self.VWAP)
                                       / (self.OPEN - ((self.HIGH + self.LOW) / 2))), M), N)) * -1)

    def alpha_088(self, N=20):
        # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
        return (self.CLOSE - DELAY(self.CLOSE, 20)) / DELAY(self.CLOSE, 20) * 100

    def alpha_089(self, N=13, M=27):
        # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
        return 2 * (SMA(self.CLOSE, N, 2) - SMA(self.CLOSE, M, 2) - SMA(SMA(self.CLOSE, N, 2) - SMA(self.CLOSE, M, 2),
                                                                        10, 2))

    def alpha_090(self, N=5):
        # ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
        return (RANK(CORR(RANK(self.VWAP), RANK(self.VOLUME), N)) * -1)

    def alpha_091(self, N=5, M=40):
        # ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)        #################
        return ((RANK((self.CLOSE - MAX(self.CLOSE, N))
                      ) * RANK(CORR((MEAN(self.VOLUME, M)), self.LOW, N))) * -1)

    def alpha_092(self, P=0.35):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        return (MAX(RANK(DECAYLINEAR(DELTA(((self.CLOSE * P) + (self.VWAP * (1 - P))), 2), 3)),
                    TSRANK(DECAYLINEAR(ABS(CORR((MEAN(self.VOLUME, 180)), self.CLOSE, 13)), 5), 15)) * -1)

    def alpha_093(self, N=20):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
        return -SUM(
            IFELSE(self.OPEN >= DELAY(self.OPEN, 1), 0, MAX((self.OPEN - self.LOW), (self.OPEN - DELAY(self.OPEN, 1)))),
            N)  # LV -1

    def alpha_094(self, N=30):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
        return SUM(IFELSE(self.CLOSE > DELAY(self.CLOSE, 1),
                          self.VOLUME,
                          IFELSE(self.CLOSE < DELAY(self.CLOSE, 1),
                                 -self.VOLUME,
                                 0)), N)

    def alpha_095(self, N=20):
        # STD(AMOUNT,20)
        return -STD(self.AMOUNT, 20)  # LVP -1

    def alpha_096(self, N=9):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
        return SMA(SMA((self.CLOSE - TSMIN(self.LOW, N)) / (TSMAX(self.HIGH, N) - TSMIN(self.LOW, N)) * 100, 3, 1), 3,
                   1)

    def alpha_097(self, N=10):
        # STD(VOLUME,10)
        return STD(self.VOLUME, N)

    def alpha_098(self, N=100, P=0.05):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)|((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))
        #  ?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        return IFELSE(((DELTA((SUM(self.CLOSE, N) / N), N) / DELAY(self.CLOSE, N)) < P)
                      | ((DELTA((SUM(self.CLOSE, N) / N), N) / DELAY(self.CLOSE, N)) == P),
                      -1 * (self.CLOSE - TSMIN(self.CLOSE, N)),
                      -1 * DELTA(self.CLOSE, 3))

    def alpha_099(self, N=5):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
        return (-1 * RANK(COVIANCE(RANK(self.CLOSE), RANK(self.VOLUME), N)))

    def alpha_100(self, N=20):
        # STD(VOLUME,20)
        return STD(self.VOLUME, 20)
    #--- 101-150
    def alpha_101(self, N=15, M=37):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1)
        return ((RANK(CORR(self.CLOSE, SUM(MEAN(self.VOLUME, 2 * N), M), N))
                 < RANK(CORR(RANK(((self.HIGH * 0.1) + (self.VWAP * 0.9))), RANK(self.VOLUME), 11))) * -1)

    def alpha_102(self, N=6):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
        return SMA(MAX(self.VOLUME - DELAY(self.VOLUME, 1), 0), N, 1) / SMA(ABS(self.VOLUME - DELAY(self.VOLUME, 1)), N,
                                                                            1) * 100

    def alpha_103(self, N=20):
        # ((20-LOWDAY(LOW,20))/20)*100
        return ((N - LOWDAY(self.LOW, N)) / N) * 100

    def alpha_104(self, N=5):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
        return (-1 * (DELTA(CORR(self.HIGH, self.VOLUME, N), N) * RANK(STD(self.CLOSE, 4 * N))))

    def alpha_105(self, N=10):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
        return (-1 * CORR(RANK(self.OPEN), RANK(self.VOLUME), N))

    def alpha_106(self, N=20):
        # CLOSE-DELAY(CLOSE,20)
        return self.CLOSE - DELAY(self.CLOSE, N)

    def alpha_107(self, N):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
        return (((-1 * RANK((self.OPEN - DELAY(self.HIGH, N)))) * RANK((self.OPEN - DELAY(self.CLOSE, N)))) * RANK(
            (self.OPEN - DELAY(self.LOW, 1))))

    def alpha_108(self, N=120):
        # ((RANK((HIGH-MIN(HIGH,2)))**RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
        return ((RANK((self.HIGH - MIN(self.HIGH, 2))) ** RANK(CORR((self.VWAP), (MEAN(self.VOLUME, N)), 6))) * -1)

    def alpha_109(self, N=10):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        return SMA(self.HIGH - self.LOW, N, 2) / SMA(SMA(self.HIGH - self.LOW, N, 2), N, 2)

    def alpha_110(self, N=20):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
        return SUM(MAX(0, self.HIGH - DELAY(self.CLOSE, 1)), N) / SUM(MAX(0, DELAY(self.CLOSE, 1) - self.LOW), N) * 100

    def alpha_111(self, N):
        # SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
        return (SMA(self.VOLUME * ((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE))
                    / (self.HIGH - self.LOW), 11, 2) - SMA(
            self.VOLUME * ((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE))
            / (self.HIGH - self.LOW), 4, 2))

    def alpha_112(self, N=12):
        # (SUM(IFELSE(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)
        #  -SUM(IFELSE(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))
        # /(SUM(IFELSE(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)
        #   +SUM(IFELSE(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))*100
        temp1 = self.CLOSE - DELAY(self.CLOSE, 1)
        temp1 = SUM(IFELSE(temp1 > 0, temp1, 0), N)
        temp2 = SUM(IFELSE(temp1 < 0, ABS(temp1), 0), N)
        return (temp1 - temp2) / (temp1 + temp2) * 100

    def alpha_113(self, N=20, M=5):
        # (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),
        # SUM(CLOSE, 20), 2))))
        return (-1 * ((RANK((SUM(DELAY(self.CLOSE, 5), N) / N))
                       * CORR(self.CLOSE, self.VOLUME, 2))
                      * RANK(CORR(SUM(self.CLOSE, M), SUM(self.CLOSE, N), 2))))

    def alpha_114(self, N=5):
        # ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /
        # (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
        return ((RANK(DELAY(((self.HIGH - self.LOW) / (SUM(self.CLOSE, N) / N)), 2)) * RANK(RANK(self.VOLUME)))
                / (((self.HIGH - self.LOW) / (SUM(self.CLOSE, N) / N)) / (self.VWAP - self.CLOSE)))

    def alpha_115(self, N=30):
        # (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))**RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))
        return (RANK(CORR(((self.HIGH * 0.9) + (self.CLOSE * 0.1)),
                          MEAN(self.VOLUME, N), 10)) ** RANK(
            CORR(TSRANK(((self.HIGH + self.LOW) / 2), 4), TSRANK(self.VOLUME, 10), 7)))

    def alpha_116(self, N=20):
        # REGBETA(CLOSE,SEQUENCE,20)
        return REGBETA(self.CLOSE, SEQUENCE, N)

    def alpha_117(self, N=16):
        # ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
        return ((TSRANK(self.VOLUME, 2 * N) * (1 - TSRANK(((self.CLOSE + self.HIGH) - self.LOW), N))) * (
                    1 - TSRANK(self.VWAP, 2 * N)))

    def alpha_118(self, N=20):
        # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
        return SUM(self.HIGH - self.OPEN, N) / SUM(self.OPEN - self.LOW, N) * 100

    def alpha_119(self, N=5, M=15):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        return (RANK(DECAYLINEAR(CORR(self.VWAP, SUM(MEAN(self.VOLUME, N), 26), 5), 7))
                - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(self.OPEN), RANK(MEAN(self.VOLUME, M)), 21), 9), 7), 8)))

    def alpha_120(self, N):
        # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
        return (RANK((self.VWAP - self.CLOSE)) / RANK((self.VWAP + self.CLOSE)))

    def alpha_121(self, N=12, M=20, K=60):
        # ((RANK((VWAP - MIN(VWAP, 12)))**TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)
        return ((RANK((self.VWAP - MIN(self.VWAP, N))) ** TSRANK(
            CORR(TSRANK(self.VWAP, M), TSRANK(MEAN(self.VOLUME, K), 2), 18), 3)) * -1)

    def alpha_122(self, N=13):
        # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)) / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        temp = SMA(SMA(SMA(LOG(self.CLOSE), N, 2), N, 2), N, 2)
        return (temp - DELAY(temp, 1)) / DELAY(temp, 1)

    def alpha_123(self, N=60, M=20):
        # ((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        return ((RANK(CORR(SUM(((self.HIGH + self.LOW) / 2), M), SUM(MEAN(self.VOLUME, N), M), 9))
                 < RANK(CORR(self.LOW, self.VOLUME, 6))) * -1)

    def alpha_124(self, N=30):
        # (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        return (self.CLOSE - self.VWAP) / DECAYLINEAR(RANK(TSMAX(self.CLOSE, N)), 2)

    def alpha_125(self, N=80):
        # (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        return (RANK(DECAYLINEAR(CORR((self.VWAP), MEAN(self.VOLUME, N), 17), 20))
                / RANK(DECAYLINEAR(DELTA(((self.CLOSE * 0.5) + (self.VWAP * 0.5)), 3), 16)))

    def alpha_126(self, N):
        # (CLOSE + HIGH + LOW) / 3
        return -(self.CLOSE + self.HIGH + self.LOW) / 3  # LVP -1

    def alpha_128(self, N=14):
        # 100-(100/(1+SUM(IFELSE((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)
        #           /SUM(IFELSE((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)))
        temp1 = (self.HIGH + self.LOW + self.CLOSE) / 3
        temp2 = DELAY(temp1, 1)
        return (100 - (100 / (1 + SUM(IFELSE(temp1 > temp2, temp1 * self.VOLUME, 0), N)
                              / SUM(IFELSE(temp1 < temp2, temp1 * self.VOLUME, 0), N))))

    def alpha_129(self, N=12):
        # SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        return -SUM(IFELSE(self.CLOSE - DELAY(self.CLOSE, 1) < 0, ABS(self.CLOSE - DELAY(self.CLOSE, 1)), 0),
                    N)  # LVP -1

    def alpha_130(self, N=40, M=10, K=3):
        # (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10))
        # / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        return (RANK(DECAYLINEAR(CORR(((self.HIGH + self.LOW) / 2), MEAN(self.VOLUME, N), 9), M)) /
                RANK(DECAYLINEAR(CORR(RANK(self.VWAP), RANK(self.VOLUME), 7), K)))

    def alpha_131(self, N=50, M=18):
        # (RANK(DELTA(VWAP, 1))**TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
        return (RANK(DELTA(self.VWAP, 1)) ** TSRANK(CORR(self.CLOSE, MEAN(self.VOLUME, N), M), M))

    def alpha_132(self, N=20):
        # MEAN(AMOUNT, 20)
        return -MEAN(self.AMOUNT, N)  # LVP -1

    def alpha_133(self, N=20):
        # ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        return ((N - HIGHDAY(self.HIGH, N)) / N) * 100 - ((N - LOWDAY(self.LOW, N)) / N) * 100

    def alpha_134(self, N=12):
        # (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        return (self.CLOSE - DELAY(self.CLOSE, N)) / DELAY(self.CLOSE, N) * self.VOLUME

    def alpha_135(self, N=20):
        # SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        return SMA(DELAY(self.CLOSE / DELAY(self.CLOSE, N), 1), N, 1)

    def alpha_136(self, N=3, M=10):
        # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        return ((-1 * RANK(DELTA(self.VWAP, N))) * CORR(self.OPEN, self.VOLUME, M))

    def alpha_137(self, N):
        # (16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))
        #  /(IFELSE(ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)),
        #           ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4,
        #           IFELSE(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)),
        #                  ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4,
        #                  ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
        #  *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))))
        DELAY1 = DELAY(self.CLOSE, 1)
        DELAY2 = DELAY(self.LOW, 1)
        temp1 = ABS(self.HIGH - DELAY1)
        temp2 = ABS(self.LOW - DELAY1)
        temp3 = ABS(self.HIGH - DELAY2)
        temp5 = ABS(DELAY1 - DELAY(self.OPEN, 1)) / 4
        temp4 = (temp1 + temp2) / 2 + temp5
        return (16 * (self.CLOSE - DELAY1 + (self.CLOSE - self.OPEN) / 2 + DELAY1 - DELAY(self.OPEN, 1))
                / IFELSE((temp1 > temp2) & (temp1 > temp3), temp4,
                         IFELSE((temp2 < temp3) & (temp2 < temp1), temp4, temp3 + temp5))
                * MAX(temp1, temp2))

    def alpha_138(self, N):
        # ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        return ((RANK(DECAYLINEAR(DELTA((((self.LOW * 0.7) + (self.VWAP * 0.3))), 3), 20))
                 - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(self.LOW, 8),
                                                  TSRANK(MEAN(self.VOLUME, 60), 17), 5), 19), 16), 7)) * -1)

    def alpha_139(self, N=10):
        # (-1 * CORR(OPEN, VOLUME, 10))
        return (-1 * CORR(self.OPEN, self.VOLUME, N))

    def alpha_140(self, N=8, M=60):
        # MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)),
        #     TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
        return MIN(RANK(DECAYLINEAR(((RANK(self.OPEN) + RANK(self.LOW)) - (RANK(self.HIGH) + RANK(self.CLOSE))), N)),
                   TSRANK(DECAYLINEAR(CORR(TSRANK(self.CLOSE, N), TSRANK(MEAN(self.VOLUME, M), 20), N), 7), 3))

    def alpha_141(self, N=15):
        # (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        return (RANK(CORR(RANK(self.HIGH), RANK(MEAN(self.VOLUME, N)), 9)) * -1)

    def alpha_142(self, N=5):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        return (((-1 * RANK(TSRANK(self.CLOSE, 2 * N)))
                 * RANK(DELTA(DELTA(self.CLOSE, 1), 1)))
                * RANK(TSRANK((self.VOLUME / MEAN(self.VOLUME, 4 * N)), N)))

    def alpha_144(self, N=20):
        # SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        return (SUMIF(ABS(self.CLOSE / DELAY(self.CLOSE, 1) - 1) / self.AMOUNT, N, self.CLOSE < DELAY(self.CLOSE, 1))
                / COUNT(self.CLOSE < DELAY(self.CLOSE, 1), N))

    def alpha_145(self, N=9, M=26, K=12):
        # (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        return (MEAN(self.VOLUME, N) - MEAN(self.VOLUME, M)) / MEAN(self.VOLUME, K) * 100

    def alpha_146(self, N=61, M=20):
        # (MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)
        #  *((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))
        #  /SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)
        #        -((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60) )
        # 原因子逻辑有问题，少了第三行第一个sma的参数，而且逻辑中有明显漏洞，例如:ret - (ret -sma(ret,61,2))
        # 这里随意改写一下。以后再看
        temp2 = SMA(self.RET, N, 2)
        temp1 = self.RET - temp2
        return MEAN(temp1, M) * temp1 / (temp2 ** 2)

    def alpha_147(self, N=12):
        # REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
        return REGBETA(MEAN(self.CLOSE, N), SEQUENCE(N))

    def alpha_148(self, N=60, M=9, K=14):
        # ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        return ((RANK(CORR((self.OPEN), SUM(MEAN(self.VOLUME, N), M), 6)) < RANK(
            (self.OPEN - TSMIN(self.OPEN, K)))) * -1)

    def alpha_150(self, N):
        # (CLOSE + HIGH + LOW)/3 * VOLUME
        return -(self.CLOSE + self.HIGH + self.LOW) / 3 * self.VOLUME  # LVP
    
    #--- 151-200
    def alpha_151(self, N=20):
        # SMA(CLOSE-DELAY(CLOSE,20),20,1)
        return SMA(self.CLOSE - DELAY(self.CLOSE, N), N, 1)

    def alpha_152(self, N=9, M=12, K=26):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-
        #     MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
        return SMA(MEAN(DELAY(SMA(DELAY(self.CLOSE / DELAY(self.CLOSE, N), 1), N, 1), 1), M)
                   - MEAN(DELAY(SMA(DELAY(self.CLOSE / DELAY(self.CLOSE, N), 1), N, 1), 1), K), N, 1)

    def alpha_153(self, N=3):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
        return (MEAN(self.CLOSE, N) + MEAN(self.CLOSE, 2 * N) + MEAN(self.CLOSE, 3 * N) + MEAN(self.CLOSE, 4 * N)) / 4

    def alpha_154(self, N=16, M=18):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
        return (((self.VWAP - MIN(self.VWAP, N))) < (CORR(self.VWAP, MEAN(self.VOLUME, 180), M)))

    def alpha_155(self, N=13, M=27, K=10):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
        temp1 = SMA(self.VOLUME, N, 2) - SMA(self.VOLUME, M, 2)
        return -(temp1 - SMA(temp1, K, 2))  # LVP -1

    def alpha_156(self, N=5, M=3, P=0.15):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1)
        return (MAX(RANK(DECAYLINEAR(DELTA(self.VWAP, N), M)),
                    RANK(DECAYLINEAR(((DELTA(((self.OPEN * P) + (self.LOW * (1 - P))), 2)
                                       / ((self.OPEN * P) + (self.LOW * (1 - P)))) * -1), M))) * -1)

    def alpha_157(self, N=5):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5))
        return (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((self.CLOSE - 1), N))))), 2), 1)))), 1), N)
                + TSRANK(DELAY((-1 * self.VWAP), 6), N))

    def alpha_158(self, N):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        temp = SMA(self.CLOSE, 15, 2)
        return ((self.HIGH - temp) - (self.LOW - temp)) / self.CLOSE

    def alpha_159(self, N=6):
        # (((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24
        #   +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24
        #   +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100
        #  /(6*12+6*24+12*24))
        temp1 = MIN(self.LOW, DELAY(self.CLOSE, 1))
        temp2 = MAX(self.HIGH, DELAY(self.CLOSE, 1)) - temp1
        return (((self.CLOSE - SUM(temp1, N)) / SUM(temp2, N) * 8 * (N ** 2)
                 + (self.CLOSE - SUM(temp1, 2 * N)) / SUM(temp2, 2 * N) * 4 * (N ** 2)
                 + (self.CLOSE - SUM(temp1, 4 * N)) / SUM(temp2, 4 * N) * 4 * (N ** 2)) * 100 / (14 * (N ** 2)))

    def alpha_160(self, N=20):
        # SMA(IFELSE(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)
        # LVP -1
        return -SMA(IFELSE(self.CLOSE <= DELAY(self.CLOSE, 1), STD(self.CLOSE, N), 0), N, 1)

    def alpha_161(self, N=12):
        # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
        # LVP -1
        return -MEAN(MAX(MAX((self.HIGH - self.LOW), ABS(DELAY(self.CLOSE, 1) - self.HIGH)),
                         ABS(DELAY(self.CLOSE, 1) - self.LOW)), N)

    def alpha_162(self, N=12):
        # ((SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #  /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        #  -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #       /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
        # /(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #       /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)
        #   -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        #        /SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)))
        temp1 = self.CLOSE - DELAY(self.CLOSE, 1)
        temp4 = SMA(MAX(temp1, 0), N, 1)
        temp5 = SMA(ABS(temp1), N, 1)
        temp6 = temp4 / temp5 * 100
        return (temp6 - MIN(temp6, N)) / (MAX(temp6, N) - MIN(temp6, N))

    def alpha_163(self, N):
        # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
        return RANK(((((-1 * self.VWAP) * MEAN(self.VOLUME, 20)) * self.VWAP) * (self.HIGH - self.CLOSE)))

    def alpha_164(self, N=12, M=13):
        # SMA((IFELSE((CLOSE>DELAY(CLOSE,1)),1/(CLOSE-DELAY(CLOSE,1)),1)
        #      -MIN(IFELSE((CLOSE>DELAY(CLOSE,1)),1/(CLOSE-DELAY(CLOSE,1)),1),12))/(HIGH-LOW)*100,13,2)
        temp1 = self.CLOSE - DELAY(self.CLOSE, 1)
        temp2 = IFELSE(temp1 > 0, 1 / temp1, 1)
        return SMA((temp2 - MIN(temp2, N)) / (self.HIGH - self.LOW) * 100, M, 2)

    def alpha_166(self, N=20):
        # 原公式有错误，随便改了改
        # -20*(20-1)**1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)
        # /((20-1)*(20-2)*(SUM(MEAN(CLOSE/DELAY(CLOSE,1),20)**2,20))**1.5)

        return (-N * (N - 1) ** 1.5 * SUM(self.RET - MEAN(self.RET, N), N)
                / ((N - 1) * (N - 2) * (SUM(MEAN(self.CLOSE / DELAY(self.CLOSE, 1), N) ** 2, N)) ** 1.5))

    def alpha_167(self, N=12):
        # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
        temp = self.CLOSE - DELAY(self.CLOSE, 1)
        return -SUM(IFELSE(temp > 0, temp, 0), N)

    def alpha_168(self, N=20):
        # (-1*VOLUME/MEAN(VOLUME,20))
        return (-1 * self.VOLUME / MEAN(self.VOLUME, N))

    def alpha_169(self, N=9, M=12, K=26):
        # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
        temp = DELAY(SMA(self.CLOSE - DELAY(self.CLOSE, 1), N, 1), 1)
        return SMA(MEAN(temp, M) - MEAN(temp, K), 10, 1)

    def alpha_170(self, N=20, M=5):
        # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /
        # 5))) - RANK((VWAP - DELAY(VWAP, 5))))
        return ((((RANK((1 / self.CLOSE)) * self.VOLUME) / MEAN(self.VOLUME, N))
                 * ((self.HIGH * RANK((self.HIGH - self.CLOSE)))
                    / (SUM(self.HIGH, M) / M)))
                - RANK((self.VWAP - DELAY(self.VWAP, M))))

    def alpha_171(self, N):
        # ((-1 * ((LOW - CLOSE) * (OPEN**5))) / ((CLOSE - HIGH) * (CLOSE**5)))
        return ((-1 * ((self.LOW - self.CLOSE) * (self.OPEN ** 5))) / ((self.CLOSE - self.HIGH) * (self.CLOSE ** 5)))

    def alpha_172(self, N=14, M=6):
        # MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #          -SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))
        #      /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #        +SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14)) *100,6)
        temp1 = (self.LD > 0) & (self.LD > self.HD)
        temp2 = (self.HD > 0) & (self.LD > self.HD)
        temp3 = SUM(IFELSE(temp1, self.LD, 0), N) * 100 / SUM(self.TR, N)
        temp4 = SUM(IFELSE(temp2, self.HD, 0), N) * 100 / SUM(self.TR, N)
        return MEAN(ABS(temp3 - temp4) / (temp3 + temp4) * 100, M)

    def alpha_173(self, N=13):
        # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
        temp = SMA(self.CLOSE, N, 2)
        # LVP -1
        return -(3 * temp - 2 * SMA(temp, N, 2) + SMA(SMA(SMA(LOG(self.CLOSE), N, 2), N, 2), N, 2))

    def alpha_174(self, N=20):
        # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        return -SMA(IFELSE(self.CLOSE > DELAY(self.CLOSE, 1), STD(self.CLOSE, N), 0), N, 1)  # LVP -1

    def alpha_175(self, N=6):
        # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
        # LVP -1
        tmp = MEAN(MAX(MAX((self.HIGH - self.LOW), ABS(DELAY(self.CLOSE, 1) - self.HIGH)),
                       ABS(DELAY(self.CLOSE, 1) - self.LOW)), N)
        return -tmp

    def alpha_176(self, N=12, M=6):
        # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
        return CORR(RANK(((self.CLOSE - TSMIN(self.LOW, N)) /
                          (TSMAX(self.HIGH, N) - TSMIN(self.LOW, N)))), RANK(self.VOLUME), M)

    def alpha_177(self, N=20):
        # ((20-HIGHDAY(HIGH,20))/20)*100
        return ((N - HIGHDAY(self.HIGH, N)) / N) * 100

    def alpha_178(self, N):
        # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
        return (self.CLOSE - DELAY(self.CLOSE, 1)) / DELAY(self.CLOSE, 1) * self.VOLUME

    def alpha_179(self, N=50):
        # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
        return (RANK(CORR(self.VWAP, self.VOLUME, 4)) * RANK(CORR(RANK(self.LOW), RANK(MEAN(self.VOLUME, N)), 12)))

    def alpha_180(self, N=20, M=7):
        # ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME)))
        return IFELSE((MEAN(self.VOLUME, N) < self.VOLUME),
                      (-1 * TSRANK(ABS(DELTA(self.CLOSE, M)), 3 * N)) * SIGN(DELTA(self.CLOSE, M)),
                      (-1 * self.VOLUME))

    def alpha_184(self, N):
        # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
        return (RANK(CORR(DELAY((self.OPEN - self.CLOSE), 1), self.CLOSE, 200)) + RANK((self.OPEN - self.CLOSE)))

    def alpha_185(self, N):
        # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
        return RANK((-1 * ((1 - (self.OPEN / self.CLOSE)) ** 2)))

    def alpha_186(self, N=14, M=6):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #           -SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))
        #       /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #         +SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        #  +DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #                  -SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))
        #              /(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)
        #                +SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #

        temp1 = (self.LD > 0) & (self.LD > self.HD)
        temp2 = (self.HD > 0) & (self.LD > self.HD)
        temp3 = SUM(IFELSE(temp1, self.LD, 0), N) * 100 / SUM(self.TR, N)
        temp4 = SUM(IFELSE(temp2, self.HD, 0), N) * 100 / SUM(self.TR, N)
        temp5 = MEAN(ABS(temp3 - temp4) / (temp3 + temp4) * 100, M)
        return (temp5 + DELAY(temp5, M)) / 2

    def alpha_187(self, N=20):
        # SUM(IFELSE(OPEN<=DELAY(OPEN,1),0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
        return SUM(IFELSE(self.OPEN <= DELAY(self.OPEN, 1), 0,
                          MAX((self.HIGH - self.OPEN), (self.OPEN - DELAY(self.OPEN, 1)))), N)

    def alpha_188(self, N=11):
        # ((HIGH - LOW - SMA(HIGH-LOW, 11, 2)) / SMA(HIGH-LOW, 11, 2))*100
        temp = self.HIGH - self.LOW
        return (temp - SMA(temp, N, 2)) / SMA(temp, N, 2) * 100

    def alpha_189(self, N=6):
        # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
        # LVP -1
        return -MEAN(ABS(self.CLOSE - MEAN(self.CLOSE, N)), N)

    def alpha_190(self, N):
        # LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))**(1/20)-1),20)-1)
        #     *(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,
        #             CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1))
        #     /((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1),20))
        #       *(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,
        #               CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))**(1/20)-1))))

        temp1 = self.CLOSE / DELAY(self.CLOSE) - 1
        temp2 = (self.CLOSE / DELAY(self.CLOSE, 19)) ** (1 / 20)
        return LOG(((COUNT(temp1 > (temp2 - 1), 20) - 1) * SUMIF((temp1 - temp2 - 1) ** 2, 20, temp1 < (temp2 - 1)))
                   / ((COUNT(temp1 < (temp2 - 1), 20)) * SUMIF((temp1 - (temp2 - 1)) ** 2, 20, temp1 > (temp2 - 1))))

    def alpha_191(self, n=20, m=5):
        # (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####
        return (CORR(MEAN(self.VOLUME, n), self.LOW, m) + ((self.HIGH + self.LOW) / 2)) - self.CLOSE

    def alpha_192(self, N=26):
        # 意愿指标	BR=N日内（当日最高价－昨日收盘价）之和 / N日内（昨日收盘价－当日最低价）之和×100 n设定为26
        return SUM(self.HIGH - self.CLOSE, N) / SUM(self.CLOSE - self.LOW, N) * 100

    def alpha_193(self, N=20, M=120):
        # ARBR 因子 AR 与因子 BR 的差
        return (SUM(self.HIGH - self.OPEN, N) / SUM(self.OPEN - self.LOW, N) * 100
                - SUM(self.HIGH - self.CLOSE, N) / SUM(self.CLOSE - self.LOW, N) * 100)

    def alpha_194(self, N=20):
        # 20日收益方差
        return VAR(self.RET, N)

    def alpha_195(self, N=20):
        # 收益的20日偏度
        return SKEW(self.RET, N)

    def alpha_196(self, N=20):
        # 收益的20日峰度
        return KURT(self.RET, N)

    def alpha_197(self, N=60):
        # 60日收益方差
        return VAR(self.RET, N)

    def alpha_198(self, N=60):
        # 收益的60日偏度
        return SKEW(self.RET, N)

    def alpha_199(self, N=60):
        # 收益的60日峰度
        return KURT(self.RET, N)

    def alpha_200(self, N=20):
        # 20日夏普比率
        return SHARPE(self.RET, N)

    #--- 201-250
    def alpha_201(self, N=60):
        # 60日夏普比率
        return SHARPE(self.RET, N)

    def alpha_202(self, N=120):
        # 120日夏普比率
        return SHARPE(self.RET, N)

    def alpha_203(self, N=120):
        # 120日收益方差
        return VAR(self.RET, N)

    def alpha_204(self, N=120):
        # 收益的120日偏度
        return SKEW(self.RET, N)

    def alpha_205(self, N=120):
        # 收益的120日峰度
        return KURT(self.RET, N)

    def alpha_206(self, N=10):
        return -(self.CLOSE / DELAY(self.CLOSE, N) - 1)

    def alpha_207(self, N=20):
        return -(self.CLOSE / DELAY(self.CLOSE, N) - 1)

    def alpha_208(self, N=10):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET > 0, self.RET, 0), N) / SUM(self.RET, N)

    def alpha_209(self, N=30):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET > 0, self.RET, 0), N) / SUM(self.RET, N)

    def alpha_210(self, N=100):
        # RSI 因子，过去 K 天累计涨幅与累 计涨跌幅度的比值
        return -1 * SUM(IFELSE(self.RET > 0, self.RET, 0), N) / SUM(self.RET, N)

    def alpha_211(self, N=10):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)

    def alpha_212(self, N=30):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)

    def alpha_213(self, N=100):
        # 隔夜趋势因子
        return MEAN(self.OPEN / self.PRE_CLOSE - 1, N)

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
        return -1 * MEAN(self.CLOSE, N) / MEAN(self.CLOSE, int(2 * N)) - 1

    def alpha_221(self, N=30):
        # 快慢均线趋势因子
        return -1 * MEAN(self.CLOSE, N) / MEAN(self.CLOSE, int(2 * N)) - 1

    def alpha_222(self, N=50):
        # 快慢均线趋势因子
        return -1 * MEAN(self.CLOSE, N) / MEAN(self.CLOSE, int(2 * N)) - 1

    def alpha_223(self, N=10):
        # 日内累计振幅因子
        # MEAN((2 * (HIGH - LOW) * SIGN(CLOSE - OPEN) - (CLOSE - OPEN)) / CLOSE, N)
        return -1 * MEAN((2 * (self.HIGH - self.LOW) * SIGN(self.CLOSE - self.OPEN)
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
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c ** 2)), N)) ** 0.5
        return -1 * MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

    def alpha_226(self, N=40):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c ** 2)), N)) ** 0.5
        return -1 * MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

    def alpha_227(self, N=100):
        # 日内波动趋势因子
        # MEAN(SIGN(CLOSE - OPEN) * GK, N)
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        GK = (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c ** 2)), N)) ** 0.5
        return -1 * MEAN(SIGN(self.CLOSE - self.OPEN) * GK, N)

    def alpha_228(self, N=10):
        # 稳健动量因子
        temp = RANK(self.RET, pct=False)
        n = temp.shape[1]
        return -1 * MEAN((temp - (n + 1) / 2) / (((n + 1) * (n - 1) / 12) ** 0.5), N)

    def alpha_229(self, N=40):
        # 稳健动量因子
        temp = RANK(self.RET, pct=False)
        n = temp.shape[1]
        return -1 * MEAN((temp - (n + 1) / 2) / (((n + 1) * (n - 1) / 12) ** 0.5), N)

    def alpha_230(self, N=5):
        # GK 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        return (252 * MEAN((0.5 * ((h - l) ** 2) - (2 * np.log(2) - 1) * (c ** 2)), N)) ** 0.5

    def alpha_231(self, N=30):
        # RS 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        c = np.log(self.CLOSE) - np.log(self.OPEN)
        return (252 * MEAN(h * (h - c) - l * (l - c), N)) ** 0.5

    def alpha_232(self, N=30):
        # PK 波动率因子
        h = np.log(self.HIGH) - np.log(self.OPEN)
        l = np.log(self.LOW) - np.log(self.OPEN)
        return (252 / 4 / np.log(2) * MEAN((h - l) ** 2, N)) ** 0.5

    def alpha_238(self, N=20, M=40):
        # M>N
        if M > N:
            return self.CLOSE / DELAY(self.CLOSE, N) - self.CLOSE / DELAY(self.CLOSE, M)
        else:
            return None

    def alpha_901(self, kdj_len, fast_ema_len, slow_ema_len):
        ll = bn.move_min(self.CLOSE, kdj_len, axis=0)
        hh = bn.move_max(self.CLOSE, kdj_len, axis=0)
        rsv = 100 * (self.CLOSE - ll) / (hh - ll)
        k = rsv.ewm(alpha=1 / fast_ema_len, adjust=False).mean()
        d = k.ewm(alpha=1 / slow_ema_len, adjust=False).mean()
        j = k * 3 - d * 2
        return -j

        ll = bn.move_min(CLOSE, kdj_len, axis=0)
        hh = bn.move_max(CLOSE, kdj_len, axis=0)
        rsv = 100 * (CLOSE - ll) / (hh - ll)
        k = rsv.ewm(alpha=1 / fast_ema_len, adjust=False).mean()
        d = k.ewm(alpha=1 / slow_ema_len, adjust=False).mean()
        j = k * 3 - d * 2

    def alpha_902(self, lenn):
        '''boll_index
        '''
        std = bn.move_std(self.CLOSE, lenn, axis=0)
        ma = bn.move_mean(self.CLOSE, lenn, axis=0)
        boll_index = (self.CLOSE - ma) / (std)
        return -boll_index

    def alpha_903(self, lenn):
        '''均线发散
        '''
        ema = self.CLOSE.ewm(adjust=False, alpha=2 / (lenn + 1), ignore_na=True).mean()
        ma = bn.move_mean(self.CLOSE, lenn, axis=0)
        mad = (ema * 3 - ma * 2) / ma
        return -mad

    def alpha_903_1(self, lenn):
        '''均线发散
        '''
        ema = self.CLOSE.ewm(adjust=False, alpha=2 / (lenn + 1), ignore_na=True).mean()
        ma = bn.move_mean(self.CLOSE, lenn, axis=0)
        mad = (ema * 3 - ma * 2) / ma
        mad = mad.diff(5)
        return mad

    #--- fundamental
    def alpha_f1(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f2(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f3(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f4(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f5(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f6(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f7(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f8(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    def alpha_f9(self, file_dir, N ):
        factor = pd.read_csv(file_dir, index_col=0, parse_dates=True)
        return MEAN(factor, N)
    
    
    
#%% MAIN
if __name__ == "__main__":

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
    
    test_date = 'factorTest_20231221'
    # test_date = f'''factorTest_{datetime.today().strftime("%Y%m%d")}'''
    
    # 输出文件夹
    filepath_test_output = f'{filepath_output}{test_date}{Description}/'
    
    filepath_output_ratios_all = f'{filepath_test_output}performance_ratios{Description}.csv'
    
    #--- 测试
    '''
    这里用的是全样本测试，
    示例是用指数收益
    '''
    dailyReturn_all = retIndex
    
    # 保证收益率和因子的表头一致
    # 因为retIndex是由CLASS wts生产的，所以是一样的
    if dailyReturn_all is retMain:
        if dailyReturn_all.columns is not retIndex.columns:
            
            tradeCols = list(set(dailyReturn_all.columns) & set(retIndex.columns))
            
            tradeCols.sort()
            
            wts = WTS_factor(price, tradeCols)
            
            dailyReturn_all = retMain[tradeCols]
            
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
    factorName = 'alpha_f1'
    
    '''
    
        
    for factorName in list_factor_test:
    # for factorName in ['alpha_f1']:
       
        print(factorName)
        
        parameter_list, list_paramName, list_paramSpace = generate_paramList(factorName, df_factorTable)
        
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
                [factor, _ret, _cost] = trim_factor(factor, retIndex, dailyReturn_all, cost)
                # 调用因子处理函数
                factor = WTS_factor_handle(factor, nsigma=3)
            #---3 截面测试
                # 有交易费的测试
                _, _, ratio = factor_test_group(factor, _ret, _cost, h=hp, factorName=test_name,
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



