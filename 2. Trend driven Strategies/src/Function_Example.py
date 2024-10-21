#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase esquelo para definir la funciones a optimizar

"""


import pandas as pd
import numpy as np

from .ObjectiveFunctionBase import AbstractObjectiveFunction

#funcion necesaria para levantar los precios, pero el codigo en si no la necesita
def get_price( symbol, interval):

        filename = "{}_{}.csv".format(symbol, interval)
        print(filename)
        df_price=pd.read_csv("../klines/downloaded/"+filename).drop_duplicates()
        price = df_price[["open","close","low","high","close_time"]].drop_duplicates().set_index("close_time")
        price.index = pd.to_datetime(price.index)
        price.index += pd.Timedelta(1, unit="ms")
        price["close"] = pd.to_numeric(price["close"])
        price = price.sort_index(ascending=True)
        price.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in price.index])
        del df_price
        return price

#Funcion que necesita la clase Optimizador
class ObjectiveFunction(AbstractObjectiveFunction):
    def __init__(self,time_serie,date_init,date_final):
        self.time_serie=time_serie
        self.date_init=date_init
        self.date_final=date_final
        self.plot=False
        
        
    def set_plot(self,plot=False):
        self.plot=plot
    
    def get_returns(self,**kwargs):
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        
        sma_short=int(kwargs["short"]) 
        sma_long =int(kwargs["long"])
        
        df=pd.DataFrame()
        df["close"]=self.time_serie["close"]
        df["close_pct_change"]=self.time_serie["close"].pct_change()
        df["sma_short"]=self.time_serie["close"].rolling(sma_short).mean()
        df["sma_long"]=self.time_serie["close"].rolling(sma_long).mean()
        df["action"]=0
        df.loc[(df["sma_long"]>df["sma_short"]) ,"action"]=-1
        df.loc[(df["sma_short"]>df["sma_long"]),"action"]=1 
        df["action"]=df["action"].shift(1).replace(0,np.nan).ffill()
        
        df=df.loc[self.date_init:self.date_final]
        df["returns"]=np.where(df["action"]==1,df["close_pct_change"],0)
        df["returns"]=np.where(df["action"]!=df["action"].shift(1),df["returns"]-0.001,df["returns"])
        return df["returns"]
        
    def function_to_optimize(self,**kwargs): ##método necesario de la función para la optimización
       
        #calculo sharpe
        returns=self.get_returns(**kwargs)
        
        result=(returns.mean())/(returns.std())*np.sqrt(365*12)
        #dado que utilizo velas de 2 horas normalizo a un retorno medio anual
        #print((1+df["returns"]).cumprod().iloc[-1],df["returns"].mean(),df["returns"].std())
        
        return result
    

