#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:04:22 2020

@author: farduh
"""
from src.OptimizeStrategyParameters import OptimizeStrategyParameters
from src.Estrategia_por_par import TwoMovingAverageStrategy,ThreeMovingAverageStrategy,SlopeStrategy
from datetime import datetime
from datetime import timedelta
import pandas as pd
osp=OptimizeStrategyParameters()
comision_margin_per_day={
    "USDT":(0.05e-2),
    "BTC":(0.02e-2),
    "BNB":(0.3e-2),
    "LTC":(0.02e-2),
    "ETH":(0.0275e-2),
    "BCH":(0.02e-2),
    "ETC":(0.02e-2),
    "XRP":(0.01e-2),
    "EOS":(0.02e-2),
    "LINK":(0.025e-2)
    }
comision_margin = pd.Series(comision_margin_per_day)
comision_margin = (1+comision_margin)**(1/12)-1
weeks_info=10
bases = list(comision_margin_per_day)
quote = "USDT"
Strategies = {"BTC":TwoMovingAverageStrategy,
              "ETH":TwoMovingAverageStrategy,
              "BNB":ThreeMovingAverageStrategy,
              }
osp=OptimizeStrategyParameters()
riesgo=5
candle_minutes=60

df=pd.DataFrame(index=range(0,51),columns=Strategies)
df.loc[:,quote]=0
for riesgo in range(0,51):
    weights=osp.update_markowitz_weights(Strategies,
            quote,
            comision_margin,
            candle_minutes,
            riesgo,
            weeks_info=weeks_info)
    df.loc[riesgo]=weights.iloc[0]
    




