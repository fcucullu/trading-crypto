#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:41:07 2020

@author: farduh
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from src.OptimizeStrategyParameters import OptimizeStrategyParameters
from src.Estrategia_por_par import BBvolumeStrategy
from datetime import datetime
from datetime import timedelta
import pandas as pd
import ray
import itertools

osp=OptimizeStrategyParameters()
comision_margin_per_day={
    "USDT":(0.05e-2),
    "BTC":(0.02e-2),
    "BNB":(0.3e-2),
    "LTC":(0.02e-2),
    "ETH":(0.0275e-2),
    "LINK":(0.025e-2)
    }
comision_margin = pd.Series(comision_margin_per_day)
comision_margin = (1+comision_margin)**(1/24)-1
bases = list(comision_margin_per_day)
quotes = ["USDT","BTC"]
Strategies = [BBvolumeStrategy]
Exp_Strategies = [None]#,AccumulatedVolumeStrategy,ChandelierExitStrategy]
Base_Strategies = [None]#,TwoMovingAverageLateralStrategy,TwoMovingAverageStrategy,ThreeMovingAverageStrategy]

bases = ["ETH"]
quotes = ["BTC","USDT"]
margin_short=[0,1]

dates = []
date = datetime(2020,9,1)
while date<datetime.now():
    dates.append(date)    
    date=dates[-1]+timedelta(weeks=1)
    
to_optimize = ["return"]

ray.init(webui_host='0.0.0.0',num_cpus=8) 
@ray.remote
def update_optimal_parameters(Strategy,base,quote,final_date,base_Strategy,exp_Strategy,opt_parameter,margin_short):
    
    print(f"run_config {Strategy.__name__} , {base}{quote}, {final_date}")
    if Strategy.__name__ == "TwoStrategiesCombination" and (base_Strategy== None or exp_Strategy==None):
        return
    elif Strategy.__name__ != "TwoStrategiesCombination" and (base_Strategy!= None or exp_Strategy!=None):
        return
    
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
            init_points=400,
            set_final_date=final_date,
            num_sim=100,
            sim_lenght=3000,
            base_strategy=base_Strategy,
            exp_strategy=exp_Strategy,
            onlineADX=False,
            optimize=opt_parameter,
            margin_short=margin_short
            )

futures=[update_optimal_parameters.remote(*args) for args in itertools.product(Strategies,bases,quotes,dates,Base_Strategies,Exp_Strategies,to_optimize,margin_short) ]
ray.get(futures)
