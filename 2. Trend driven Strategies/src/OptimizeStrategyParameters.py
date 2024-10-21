#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:06:42 2020

@author: farduh
"""
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.covariance import LedoitWolf
from .candlestick import CandlestickRepository
from .simulations import MertonJumpDiffusionSimulator,PriceWorker, PriceExplorer, SimulateMLMertonOHLC, SimulateMLGeometricOHLC,Fitter, DistributionGenerator, FillNa
from .ModelOptimization import ModelOptimization
from .HyperParametersGenerator import BayesianHyperParametersGenerator
from .Estrategia_por_par import HoldingStrategy
from .Recomendador_v2 import Recomendador

class OptimizeStrategyParameters:
    
    def update_pair_strategy_optimal_parameters(
            self,
            Strategy,
            comision_margin,
            base,
            quote,
            candle_minutes,
            BenchmarkStrategy=HoldingStrategy,
            weeks_info=10,
            verbose=2,
            Xcapit_ratio=True,
            **kwargs,
            ):
        """
        Optimize hyperparameters in pair strategies
        """
        kwargs.setdefault("set_final_date",datetime.now())
        kwargs.setdefault("n_iter",200)
        kwargs.setdefault("init_points",100)
        kwargs.setdefault("num_sim",1000)
        kwargs.setdefault("sim_length",10000)
        kwargs.setdefault("optimize","sharpe")
        kwargs.setdefault("margin_short",0)
        date_final = kwargs["set_final_date"]
        
        try:
            df = pd.read_csv("data/initial_date_to_train.csv")
            date_init = pd.to_datetime(df["date_init"]).iloc[0]
        except:
           date_init = date_final-timedelta(weeks=weeks_info)
        #generating simulations
        simulations = self.generate_simulation(date_init,
                                             date_final,
                                             candle_minutes,
                                             base,
                                             quote,
                                             num_sim=kwargs["num_sim"],
                                             sim_length=kwargs["sim_length"])
        #Setting strategy
        strategy=Strategy(simulations,
                          base,
                          quote,
                          comision_margin,
                          date_init=simulations[0].index[300],
                          date_final=simulations[0].index[-1],
                          candle_minutes=candle_minutes,
                          **kwargs)
        
        #Setting benchmark strategy
        benchmark_strategy = BenchmarkStrategy(simulations,
                                             base,
                                             quote,
                                             comision_margin,
                                             date_init=simulations[0].index[300],
                                             date_final=simulations[0].index[-1],
                                             candle_minutes=candle_minutes,
                                             **kwargs)
        
        #Optimizing strategy
        mopt = ModelOptimization(strategy,BayesianHyperParametersGenerator)
        try:
            pd.read_csv(f"output/optimal_params_{kwargs['optimize']}_{strategy.name}_{date_final.strftime('%Y-%m-%d_%H')}_{base}{quote}_{candle_minutes}_mshort{kwargs['margin_short']}.csv")
            print('Already optimized')
            return
        except:
            validation_size = 0.2 if Xcapit_ratio else 0
            n_split = 1 if Xcapit_ratio else 0
#            optimal,validation = mopt.run_full_optimization(strategy.pbounds,
#                                       train_size=1,
#                                       validation_size=validation_size,
#                                       test_size=0,
#                                       n_split=n_split,
#                                       benchmark_returns=benchmark_strategy,
#                                       Xcapit_ratio_span=1,
#                                       n_iter=n_iter,
#                                       init_points=init_points,
#                                       verbose=2
#                                       )
            
            validation_results = mopt.validation.validate_models(
                    strategy.pbounds,simulations[0].index[300],
                    simulations[0].index[-1],validation_size,n_split=n_split,**kwargs,verbose=2)
            Xcapit_ratio,validation_results=mopt.get_xcapit_ratio(
                    validation_results,benchmark_strategy,span=1)

        
            if not os.path.exists('output'):
                os.makedirs('output')
            validation_results.to_csv(f"output/optimal_params_{kwargs['optimize']}_{strategy.name}_{date_final.strftime('%Y-%m-%d_%H')}_{base}{quote}_{candle_minutes}_mshort{kwargs['margin_short']}.csv",index=False)
            return

    def update_markowitz_weights(
            self,
            Strategies:dict,
            quote:str,
            comision_margin:pd.DataFrame,
            candle_minutes:int,
            riesgo:float,
            weeks_info=10,
            **kwargs,
            ):
        """
        Get markowitz weights
        """
        
        kwargs.setdefault("set_final_date",datetime.now())
        kwargs.setdefault("num_sim",1000)
        kwargs.setdefault("sim_length",10000)
        date_final=kwargs["set_final_date"]
        bases=list(Strategies.keys())
        date_init=date_final-timedelta(weeks=weeks_info)
        expected_returns={quote:0}
        returns=pd.DataFrame()
        for base in bases:
            #generating simulations
            simulations=self.generate_simulation(
                    date_init,
                    date_final,
                    candle_minutes,
                    base,
                    quote,
                    num_sim=kwargs["num_sim"],
                    sim_length=kwargs["sim_length"])
            print(kwargs[base])
            strategy=Strategies[base](
                    simulations,
                    base,
                    quote,
                    comision_margin,
                    date_init=simulations[0].index[400],
                    date_final=simulations[0].index[-1],
                    candle_minutes=candle_minutes,
                    optimize="return",
                    **kwargs[base]) 
#            datetimes=[]
            if not os.path.exists('output'):
                os.makedirs('output')
#            for file in glob(f"output/optimal_params_{strategy.name}_*_{base}{quote}_{candle_minutes}.csv"):
#                file=file.split('_')
#                dates=datetime.strptime(f'{file[3]}_{file[4]}','%Y-%m-%d_%H')
#                datetimes.append(dates)
#            optimization_date=max(datetimes)
            optimization_date=date_final
            optimal_parameters=pd.read_csv(
                    f"output/optimal_params_{strategy.name}_{optimization_date.strftime('%Y-%m-%d_%H')}_{base}{quote}_{candle_minutes}.csv")
            optimal_parameters=optimal_parameters.loc[0].to_dict()
            strategy_return=strategy.function_to_optimize(**optimal_parameters)
            expected_returns.update({base:strategy_return})
            #Covariance
            historical_prices=pd.DataFrame()
            historical_prices=self.get_prices(
                    base,quote,candle_minutes,date_init,date_final)
            historical_prices=[historical_prices]

            strategy=Strategies[base](historical_prices,
                          base,
                          quote,
                          comision_margin,
                          date_init=simulations[0].index[400],
                          date_final=simulations[0].index[-1],
                          candle_minutes=candle_minutes,
                          **kwargs[base])
            returns.loc[:,base]=strategy.get_returns(0,**optimal_parameters)
        returns.loc[:,quote]=0
        expected_returns=pd.DataFrame([expected_returns])
        covariances=returns.rolling(len(returns)).cov().dropna()
        corr=returns.rolling(len(returns)).corr().dropna()
        covariances.loc[:,:]=LedoitWolf().fit(returns.dropna()).covariance_
        expected_returns.index=[covariances.index[-1][0]]
        periods = (60*24*365)/candle_minutes
        weights=Recomendador().generar_recomendacion(
                expected_returns,
                covariances,riesgo=riesgo,periods=periods)
        return weights, expected_returns, covariances, corr

    def generate_simulation(self,
                            date_initial,
                            date_final,
                            minutes,
                            base,
                            quote,
                            num_sim=1000,
                            sim_length=10000):
        """
        Generate simulations
        """
        if not os.path.exists('samples'):
            os.makedirs('samples')
        print("Getting data")
        data=self.get_prices(base,quote,minutes,date_initial,date_final)
        periods = (60*24*365)/minutes   
        print('Analyzing homoscedasticity')
        explorer, worker = PriceExplorer(), PriceWorker()
        sample = explorer.get_sample_for_simulations_non_normal(data['close'], periods, 0.01)
        sample = worker.rebuild_tabla_by_index(data, sample)
        date_initial,date_final = sample.index[0], sample.index[-1]
        try:
            simulations = pd.read_csv(f"samples/simulation_{date_final.strftime('%Y%m%d')}-{date_initial.strftime('%Y%m%d')}_{base}{quote}_{minutes}t.csv",index_col="Unnamed: 0")
            simulations.index = pd.to_datetime(simulations.index)
            simulations_list = []
            for i in range(int(len(simulations.columns)/5)):
                df=pd.DataFrame(index = simulations.index)
                df["open"] = simulations[f"open_{i}"]
                df["close"] = simulations[f"close_{i}"]
                df["high"] = simulations[f"high_{i}"]
                df["low"] = simulations[f"low_{i}"]
                df["volume"] = simulations[f"volume_{i}"]
                simulations_list.append(df)
                del df
            print('Simulations already exist')
        except:
            print("Generating simulation")
            simulator = SimulateMLMertonOHLC(sample, periods)
            if np.isnan(simulator.sigma_jhat):
                simulator = SimulateMLGeometricOHLC(sample, periods)
            simulator.fix_mean_corrfactor(data)
            simulations_list = simulator.simulate_ohlc_n_times_with_variable_mean(np.zeros(num_sim),
                    num_sim, sim_length)
            date_index=pd.date_range(end=sample.index[-1],freq=f"{minutes}t",periods=sim_length)
            df=pd.DataFrame(index=date_index)
            i=0
            for simulations in simulations_list:
                simulations.index=date_index
                df[f"close_{i}"]=simulations["close"]
                df[f"high_{i}"]=simulations["high"]
                df[f"low_{i}"]=simulations["low"]
                df[f"open_{i}"]=simulations["open"]
                df[f"volume_{i}"]=simulations["volume"]
                i+=1
            df.to_csv(f"samples/simulation_{date_final.strftime('%Y%m%d')}-{date_initial.strftime('%Y%m%d')}_{base}{quote}_{minutes}t.csv")
        return simulations_list
    
    def get_prices(self,base,quote,minutes,date_initial,date_final):
        """
        Get prices
        """
        columns_wanted = ['open','high','low','close','volume']
        candles = CandlestickRepository.preprod_repository()
        ticker=f'{base}/{quote}'
        date_final+=timedelta(microseconds=1)
        table = candles.get_candlestick(ticker,'binance',minutes,date_initial,date_final)
        if table.empty:
            ticker=f'{quote}/{base}'
            table = candles.get_candlestick(
                    ticker,'binance',minutes,date_initial,date_final)
            data = pd.DataFrame()
            data["close"] = 10000/table["close"]
            data["open"] = 10000/table["open"]
            data["low"] = 10000/table["high"]
            data["high"] = 10000/table["low"]
            data["volume"] = table["volume"]
        else:
            data=table[columns_wanted]
            del table
        data = FillNa().fill_ohlc(data)
        return data.iloc[0:-2]