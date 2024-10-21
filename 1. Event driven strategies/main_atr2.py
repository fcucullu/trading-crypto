#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:38:45 2020

@author: fcucullu
"""

import sys
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia-fran-local')
from src.OptimizeStrategyParameters import OptimizeStrategyParameters
from src.Estrategia_por_par import TwoStrategiesCombination, AccumulatedVolumeStrategy, TwoMovingAverageStrategy, ChandelierExitStrategy, LongTraillingStopLoss
from datetime import datetime
import pandas as pd
import itertools

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
bases = ["BTC"]
quotes = ["USDT"]
#Strategies= [TwoMovingAverageStrategy]
Strategies = [TwoStrategiesCombination]
dates = [datetime(2020,1,1)]


def update_optimal_parameters(Strategy,base,quote,final_date):
    #print(f"run_config {Strategy.__name__}, {base}{quote}, {final_date}")
    if base==quote:
        return
    osp.update_pair_strategy_optimal_parameters(
            Strategy,
            comision_margin,
            base,
            quote,
            60,
            weeks_info=10,
            n_iter=200,
            init_points=100,
            set_final_date=final_date,
            num_sim=1,
            sim_length=1000,
            base_strategy=TwoMovingAverageStrategy,
            exp_strategy=LongTraillingStopLoss)
futures=[update_optimal_parameters(*args) for args in itertools.product(Strategies,bases,quotes,dates) ]

