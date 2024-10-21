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
        
    return price


def deco_price(price):
        """
        input:
            DataFrame con los precios en columnas de close, high, low
        output:
            DataFrame decorado con las variables necesarias para el algoritmo de caida fuerte
        """
    
        #Defino variables que se utilizan para calcular el stoploss
        sma_30=price["close"].rolling(30).mean().shift(1)
        price["r_30"]=price["close"].pct_change().rolling(30).mean().shift(1)
        
        diff_30l=(price["low"].shift(1)-sma_30)/sma_30
        diff_30h=(price["high"].shift(1)-sma_30)/sma_30
        
        price["std_30h"]=diff_30h.rolling(700).std()
        price["std_30l"]=diff_30l.rolling(700).std()
        
        price["sma_20"]=price["close"].shift(1).rolling(20).mean()
        price["C_1"]=price["close"].shift(1)
        
        price["sma_30h"]=price["high"].shift(1).rolling(30).mean()
        price["sma_30l"]=price["low"].shift(1).rolling(30).mean()
        
        return price

#Funcion que necesita la clase Optimizador
class Objective_Function(AbstractObjectiveFunction):
    
    def function_to_optimize(self,**kwargs): ##método necesario de la función para la optimización
        a2=kwargs["a2"]
        a3=kwargs["a3"]
        b2=kwargs["b2"]
        b3=kwargs["b3"]
        df=self.time_serie
        
        self.time_serie["limup"]= pd.eval("(1+(self.time_serie.std_30h)*(a2)+a3*(self.time_serie.C_1-self.time_serie.sma_20)/self.time_serie.sma_20)*self.time_serie.sma_30h")
        self.time_serie["limdown"]=pd.eval("(1-(self.time_serie.std_30l)*(b2)+b3*(self.time_serie.C_1-self.time_serie.sma_20)/self.time_serie.sma_20)*self.time_serie.sma_30l")
        self.time_serie["returns"]=self.get_stoploss_returns(self.time_serie.loc[self.date_init:self.date_final])
        
        return (1+self.time_serie["returns"].dropna()).cumprod().iloc[-1]
        
        
    def get_stoploss_returns(self,df):
        """
        -input: df=df[["open","low","high","close","limup","limdown"]]
            open, high, low, close son los precios usuales.
            limup y limdown limites en los cuales se realizan la compra y venta del activo
            
        -output: serie con los retornos obtenidos
        
        -nota: las señales por stoploss se ejecutan por vela, esto quiere decir que en caso de que el precio caiga y se active un stoploss, no se ejecutara otra operación en esa vela.
        
        
        """
        df_temp=df.copy()
        
        #Defino series que voy a utilizar
        
        df_temp["action"]=0
        df_temp["returns"]=0
        df_temp["sell"]=0
        df_temp["buy"]=0
        
        
        #Si el precio subió mís que el límite se activo la compra
        df_temp["buy"].loc[df_temp.high>df["limup"]]=1
        #Si el precio bajó más que el límite se activo la venta
        df_temp["sell"].loc[df_temp.low<df["limdown"]]=1
        
        #Defino la acción a realizar
        df_temp["action"].loc[((df_temp["buy"]==1))]=1
        df_temp["action"].loc[((df_temp["sell"]==1))]=-1
        
        #En caso que acción se encuentre una acción de compra y venta
        df_temp["action"].loc[((df_temp["buy"]==1) & (df_temp["sell"]==1))]=3
        df_temp["action"]=df_temp["action"].replace(0,np.nan).ffill().replace(np.nan,0)
        df_temp["action"].loc[((df_temp["buy"]==1) & (df_temp["sell"]==1))]=2
        df_temp["action"]=df_temp["action"].replace(3,0)
        df_temp["action"].loc[df_temp[df_temp["buy"]==1].index.min()]=1
        i=0
        while((df_temp["action"]==2).any() or (df_temp["action"]==-2).any()):
            df_temp["action"]=np.where((df_temp["action"]==2) | (df_temp["action"]==(-2)) ,df_temp["action"].shift(1)*(-1),df_temp["action"])
            df_temp["action"]=df_temp["action"].replace(0,np.nan).ffill().replace(np.nan,0)
            i+=1
            if i>500: # en caso que las acciones de compra y venta seguidas superen las 500 se devuelve una serie vacía
                #print("too many buy and sell orders together") debug
                return df_temp["returns"]
    
        df_temp["pct_action"]=df_temp["action"].pct_change().replace(np.nan,0)
    
        
        #Si hay compra:
        #y el valor de apertura es menor al límite puesto, se compra al precio soporte
        df_temp["returns"].loc[(df_temp["action"]==1)&(df_temp["pct_action"]!=0)& (df["limup"]>df_temp["open"])]=((df_temp["close"]-df["limup"])/df["limup"])-0.002
        #y el valor de apertura es mayor al límite puesto, se compra al precio de apertura
        df_temp["returns"].loc[(df_temp["action"]==1)&(df_temp["pct_action"]!=0) & (df["limup"]<=df_temp["open"])]=(df_temp["close"]-df_temp["open"])/(df_temp["open"])-0.002
        #Si hay venta
        #y el valor de venta es menor al precio de apertura se compra al precio de soporte
        df_temp["returns"].loc[(df_temp["action"]==(-1))&(df_temp["pct_action"]!=0)& (df["limdown"]<=df_temp["open"])]=((df_temp["limdown"]-df_temp["close"].shift(1))/df_temp["close"].shift(1))-0.002
        #y el valor de venta es mayor al precio de apertura se compra al precio de apertura
        df_temp["returns"].loc[(df_temp["action"]==(-1))&(df_temp["pct_action"]!=0)& (df["limdown"]>df_temp["open"])]=(-0.002)
        #Si estoy en períodos sin trancisiones los retornos vienen dandos por:
        #el retorno es el del par, en caso de estar comprado
    
        df_temp["returns"].loc[(df_temp["pct_action"]==0)& (df_temp["action"]==1)]=(df_temp["close"]-df_temp["close"].shift(1))/df_temp["close"].shift(1)
        
        return df_temp["returns"]


    