# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia_explosiva\src')
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Optimizador\src')
import warnings
warnings.filterwarnings("ignore")

path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\pivot_points\src'
sys.path.append(path)
from funciones_auxiliares import get_puntos_extremos
from funciones_auxiliares import *

from ObjectiveFunctionBase import AbstractObjectiveFunction

#funcion necesaria para levantar los precios, pero el codigo en si no la necesita
def _get_price(path):

        df_price=pd.read_csv(path).drop_duplicates()
        price = df_price[["open","close","low","high","volume", "close_time"]].drop_duplicates().set_index("close_time")
        price.index = pd.to_datetime(price.index)
        price.index += pd.Timedelta(1, unit="ms")
        price["close"] = pd.to_numeric(price["close"])
        price = price.sort_index(ascending=True)
        price.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in price.index])
        del df_price
        return price



#Funcion que necesita la clase Optimizador
class Objective_Function(AbstractObjectiveFunction):

    def calculate_true_range(self):
        '''
        El máximo de entre:
            * Diferencia entre el high y el low del mismo dia,
            * Diferencia entre el high de hoy menos el cierre de ayer, y
            * Diferencia entre el low de hoy menos el cierre de ayer.
        '''
        
        data = self.time_serie.copy()

        data['TR'] = 0
        
        for index in range(len(data)):
            if index != 0:
                tr1 = data["high"].iloc[index] - data["low"].iloc[index]
                tr2 = abs(data["high"].iloc[index] - data["close"].iloc[index-1])
                tr3 = abs(data["low"].iloc[index] - data["close"].iloc[index-1])
                
                true_range = round(max(tr1, tr2, tr3), 4)
                data['TR'].iloc[index] = true_range

        self.time_serie = data
        return 

    def calculate_average_true_range(self, mean):
        '''
        Para el primer ATR, calcula el promedio de los ultimos mean periodos.
        Para los siguientes ATRs, multiplica el ultimo ATR calculado por mean-1,
        le suma el ultimo TR calculado y luego divide todo por mean.
        
        '''
        self.calculate_true_range()
        
        data = self.time_serie.copy()
        
        data['ATR'] = 0
        data['ATR'].iloc[mean] = round( data["TR"][1:mean+1].rolling(window=mean).mean()[-1], 4)
        
        for index in range(mean+1, len(data)):
            
            data['ATR'].iloc[index] = (data['ATR'].iloc[index-1] * (mean-1) + data['TR'].iloc[index] ) / mean
        
        self.time_serie = data
        return
    
    def calculate_chandelier_exits(self, mean, chand_window, mult_high, mult_low):
        
        mean=int(mean)
        chand_window=int(chand_window)        
    
        self.calculate_average_true_range(mean)
        
        data = self.time_serie.copy()
        
        data["chandelier_high"] = data['close']
        data["chandelier_low"] = data['close']
        for index in range(chand_window+1, len(data)):
            
            data["chandelier_low"].iloc[index] = data["low"][index-chand_window:index].min() + mult_high * data["ATR"].iloc[index]
            data["chandelier_high"].iloc[index] = data["high"][index-chand_window:index].max() - mult_low * data["ATR"].iloc[index]
        
        self.time_serie = data
        return
    
    def get_actions(self):
        data = self.time_serie.copy()
        
        #defino la acciones a realizar
        data["action"] = 0
            #COMPRA: El precio tiene que estar por encima de las dos lineas
        data["action"] = np.where(data["close"] > data["chandelier_low"],
                                  1,
                                  data["action"])
            #VENTA: El precio tiene que estar por debajo de chand_low
        data["action"] = np.where(data["close"] < data["chandelier_high"],
                                  -1,
                                  data["action"])
        
        
        data["action"] = data["action"].shift(1) #ACA SI CONSIDERO SHIFTS DE LA BASE PARA OPERAR EN OPEN SIGUIENTE
        #saco la primer obs, ahora vacia
        data = data[1:]
        
        self.time_serie = data.copy()

    def calculate_mean_dd_between_operations(self, data, data_signals, index_max_data, index_min_data):
        
        data_signals['pp_last'] = .0
        data_signals['max_dd'] = .0
        for obs in range(1, len(data_signals)):
            
            #Identifico index entre operaciones
            index_end = data.index.get_loc(data_signals.index[obs])
            index_start = data.index.get_loc(data_signals.index[obs-1])
            
            #Encuentro nivel del PP anterior. Me fijo si debe ser Max o Min
            #Si el final es compra, tengo que encontrar el minimo anterior
            if data_signals['action'][obs] == 1:
                pp_list = [n for n in index_min_data if n >= index_start and n <= index_end]
                if len(pp_list) == 0:
                    pp_list = [index_start]
                    
                pp_prices = []
                for pp in pp_list:
                    pp_prices.append(data.iloc[pp]['close'])
                    
                index_best = pp_list[pp_prices.index(min(pp_prices))]
                    
                data_signals['pp_last'][obs] = data.close[index_best]
                data_signals['max_dd'][obs] =  (data.close[index_best] / data_signals.close[obs] -1) * 100
    
                
            #Si el final es venta, tengo que encontrar el maximo anterior    
            elif data_signals['action'][obs] == -1:
                pp_list = [n for n in index_max_data if n >= index_start and n <= index_end]
                if len(pp_list) == 0:
                    pp_list = [index_start]
                    
                pp_prices = []
                for pp in pp_list:
                    pp_prices.append(data.iloc[pp]['close'])
                    
                index_best = pp_list[pp_prices.index(min(pp_prices))]
                    
                data_signals['pp_last'][obs] = data.close[index_best]
                data_signals['max_dd'][obs] =  (data_signals.close[obs] / data.close[index_best] -1) * 100
                    
                
        mean_dd = data_signals[data_signals['max_dd'] < 0]['max_dd'].mean()
    
        return mean_dd
        
 
    
    def define_pp(self, index_max_data, index_min_data):
        
         self.index_max_data = index_max_data
         self.index_min_data = index_min_data
      
    
    def function_to_optimize(self,
                            mult_high,
                            mult_low,
                            chand_window,
                            mean,
                            **kwargs): ##método necesario de la función para la optimización

        
        mean=int(mean)
        chand_window=int(chand_window)
              
        #información a utilizar
        self.calculate_chandelier_exits(mean, chand_window, mult_high, mult_low)
               
        self.get_actions()
                
        data = self.time_serie.copy()
        
        
        #defino df2 solo con acciones de compra y venta
        data_signals = data[data['action'] != 0]
        data_signals = data_signals[data_signals["action"] != data_signals["action"].shift(1)] #le saca operaciones repetidas
          
        mean_dd = self.calculate_mean_dd_between_operations(data, data_signals, self.index_max_data, self.index_min_data)
    
    
        return mean_dd
    
    
    
    
def graph_signals(data, data_signals):
    
    import matplotlib.pyplot as plt
 
    data_signals = data_signals[data_signals.index < data.index[-1]]
    data_signals = data_signals[data_signals.index > data.index[0]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(data['close'], linewidth = 2, color = 'black')
    plt.plot(data['chandelier_high'], linewidth = 1, color = 'green')
    plt.plot(data['chandelier_low'], linewidth = 1, color = 'red')
    plt.legend(['Precio','High','Low'], prop={'size': 20})
    plt.title('Precios close + Chand_Exits')
    plt.xlabel('Observaciones')
    plt.ylabel('Precio close')
    
    for obs in range(len(data_signals['action'])):
        if data_signals['action'][obs] == 1:
            plt.scatter(data_signals.index[obs], data_signals.close[obs], s=100, color='g')
        elif data_signals['action'][obs] == -1:
            plt.scatter(data_signals.index[obs], data_signals.close[obs], s=100, color='r')
        
    plt.show()
    

def graph_signals_and_pivot_points(data, data_signals):
    
    path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\pivot_points\src'
    sys.path.append(path)
    from funciones_auxiliares import get_puntos_extremos
    import matplotlib.pyplot as plt

    
    data_signals = data_signals[data_signals.index < data.index[-1]]
    data_signals = data_signals[data_signals.index > data.index[0]]
    
    index_max_data = get_puntos_extremos(data, 'close', 10, 'maximos')
    index_min_data = get_puntos_extremos(data, 'close', 10, 'minimos')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(data['close'], linewidth = 2, color = 'black')
    plt.plot(data['chandelier_high'], linewidth = 1, color = 'green')
    plt.plot(data['chandelier_low'], linewidth = 1, color = 'red')
    plt.legend(['Precio','High','Low'], prop={'size': 20})
    plt.title('Precios close + Chand_Exits')
    plt.xlabel('Observaciones')
    plt.ylabel('Precio close')
    
    for obs in range(len(data_signals['action'])):
        if data_signals['action'][obs] == 1:
            plt.scatter(data_signals.index[obs], data_signals.close[obs], s=100, color='g')
        elif data_signals['action'][obs] == -1:
            plt.scatter(data_signals.index[obs], data_signals.close[obs], s=100, color='r')
        
    for obs in index_max_data:
        plt.scatter(data.index[obs], data.close[obs], s=25, marker='s', color='b')
        
    for obs in index_min_data:
        plt.scatter(data.index[obs], data.close[obs], s=25, marker='s',color='b')
        
    plt.show()
    
    
    
    
    
    

    
    
    
    
    
    
    