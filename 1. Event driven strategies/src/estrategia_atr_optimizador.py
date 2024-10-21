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

    def calcula_true_range(self):
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

    def calcula_average_true_range(self, media):
        '''
        Para el primer ATR, calcula el promedio de los ultimos MEDIA periodos.
        Para los siguientes ATRs, multiplica el ultimo ATR calculado por MEDIA-1,
        le suma el ultimo TR calculado y luego divide todo por MEDIA.
        
        '''
        self.calcula_true_range()
        
        data = self.time_serie.copy()
        
        data['ATR'] = 0
        data['ATR'].iloc[media] = round( data["TR"][1:media+1].rolling(window=media).mean()[-1], 4)
        
        for index in range(media+1, len(data)):
            
            data['ATR'].iloc[index] = (data['ATR'].iloc[index-1] * (media-1) + data['TR'].iloc[index] ) / media
        
        self.time_serie = data
        return
    
    def calcula_chandelier_exits(self, media, chand_window, mult_high, mult_low):
        
        media=int(media)
        chand_window=int(chand_window)        
    
        self.calcula_average_true_range(media)
        
        data = self.time_serie.copy()
        
        data["chandelier_high"] = data['close']
        data["chandelier_low"] = data['close']
        for index in range(chand_window+1, len(data)):
            
            data["chandelier_low"].iloc[index] = data["low"][index-chand_window:index].min() + mult_high * data["ATR"].iloc[index]
            data["chandelier_high"].iloc[index] = data["high"][index-chand_window:index].max() - mult_low * data["ATR"].iloc[index]
        
        self.time_serie = data
        return
    
    def calcula_senales(self):
        data = self.time_serie.copy()
        
        #defino la acciones a realizar
        data["action"] = 0
            #COMPRA: El precio tiene que estar por encima de las dos lineas
        data["action"] = np.where((data["close"] > data["chandelier_low"]) & (data["close"] > data["chandelier_low"]),
                                  1,
                                  data["action"])
            #VENTA: El precio tiene que estar por debajo de chand_low
        data["action"] = np.where(data["close"] < data["chandelier_low"],
                                  -1,
                                  data["action"])
        
        
        data["action"] = data["action"].shift(1) #ACA SI CONSIDERO SHIFTS DE LA BASE PARA OPERAR EN OPEN SIGUIENTE
        #saco la primer obs, ahora vacia
        data = data[1:]
        
        self.time_serie = data.copy()

    def calcula_mean_dd_entre_operaciones(self, data, data_senales, indice_max_data, indice_min_data):
        
        data_senales['pp_anterior'] = .0
        data_senales['max_dd'] = .0
        for obs in range(1, len(data_senales)):
            
            #Identifico index entre operaciones
            index_en_data_fin = data.index.get_loc(data_senales.index[obs])
            index_en_data_ini = data.index.get_loc(data_senales.index[obs-1])
            
            #Encuentro nivel del PP anterior. Me fijo si debe ser Max o Min
            #Si el final es compra, tengo que encontrar el minimo anterior
            if data_senales['action'][obs] == 1:
                lista_de_pp = [n for n in indice_min_data if n >= index_en_data_ini and n <= index_en_data_fin]
                if len(lista_de_pp) == 0:
                    lista_de_pp = [index_en_data_ini]
                    
                precios_de_pp = []
                for pp in lista_de_pp:
                    precios_de_pp.append(data.iloc[pp]['close'])
                    
                index_optimo = lista_de_pp[precios_de_pp.index(min(precios_de_pp))]
                    
                data_senales['pp_anterior'][obs] = data.close[index_optimo]
                data_senales['max_dd'][obs] =  (data.close[index_optimo] / data_senales.close[obs] -1) * 100
    
                
            #Si el final es venta, tengo que encontrar el maximo anterior    
            elif data_senales['action'][obs] == -1:
                lista_de_pp = [n for n in indice_max_data if n >= index_en_data_ini and n <= index_en_data_fin]
                if len(lista_de_pp) == 0:
                    lista_de_pp = [index_en_data_ini]
                    
                precios_de_pp = []
                for pp in lista_de_pp:
                    precios_de_pp.append(data.iloc[pp]['close'])
                    
                index_optimo = lista_de_pp[precios_de_pp.index(min(precios_de_pp))]
                    
                data_senales['pp_anterior'][obs] = data.close[index_optimo]
                data_senales['max_dd'][obs] =  (data_senales.close[obs] / data.close[index_optimo] -1) * 100
                    
                
        mean_dd = data_senales[data_senales['max_dd'] < 0]['max_dd'].mean()
    
        return mean_dd
        
 
    
    def definir_pp(self, indice_max_data, indice_min_data):
        
         self.indice_max_data = indice_max_data
         self.indice_min_data = indice_min_data
      
    
    def function_to_optimize(self,
                            mult_high,
                            mult_low,
                            chand_window,
                            media,
                            **kwargs): ##método necesario de la función para la optimización

        
        media=int(media)
        chand_window=int(chand_window)
              
        #información a utilizar
        self.calcula_chandelier_exits(media, chand_window, mult_high, mult_low)
               
        self.calcula_senales()
                
        data = self.time_serie.copy()
        
        
        #defino df2 solo con acciones de compra y venta
        data_senales = data[data['action'] != 0]
        data_senales = data_senales[data_senales["action"] != data_senales["action"].shift(1)] #le saca operaciones repetidas
          
        mean_dd = self.calcula_mean_dd_entre_operaciones(data, data_senales, self.indice_max_data, self.indice_min_data)
    
    
        return mean_dd
    
    
    
    
def graficar_senales(data, data_senales):
    
    import matplotlib.pyplot as plt
 
    data_senales = data_senales[data_senales.index < data.index[-1]]
    data_senales = data_senales[data_senales.index > data.index[0]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(data['close'], linewidth = 2, color = 'black')
    plt.plot(data['chandelier_high'], linewidth = 1, color = 'green')
    plt.plot(data['chandelier_low'], linewidth = 1, color = 'red')
    plt.legend(['Precio','High','Low'], prop={'size': 20})
    plt.title('Precios close + Chand_Exits')
    plt.xlabel('Observaciones')
    plt.ylabel('Precio close')
    
    for obs in range(len(data_senales['action'])):
        if data_senales['action'][obs] == 1:
            plt.scatter(data_senales.index[obs], data_senales.close[obs], s=100, color='g')
        elif data_senales['action'][obs] == -1:
            plt.scatter(data_senales.index[obs], data_senales.close[obs], s=100, color='r')
        
    plt.show()
    

def graficar_senales_y_pivot_points(data, data_senales):
    
    path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\pivot_points\src'
    sys.path.append(path)
    from funciones_auxiliares import get_puntos_extremos
    import matplotlib.pyplot as plt

    
    data_senales = data_senales[data_senales.index < data.index[-1]]
    data_senales = data_senales[data_senales.index > data.index[0]]
    
    indice_max_data = get_puntos_extremos(data, 'close', 10, 'maximos')
    indice_min_data = get_puntos_extremos(data, 'close', 10, 'minimos')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(data['close'], linewidth = 2, color = 'black')
    plt.plot(data['chandelier_high'], linewidth = 1, color = 'green')
    plt.plot(data['chandelier_low'], linewidth = 1, color = 'red')
    plt.legend(['Precio','High','Low'], prop={'size': 20})
    plt.title('Precios close + Chand_Exits')
    plt.xlabel('Observaciones')
    plt.ylabel('Precio close')
    
    for obs in range(len(data_senales['action'])):
        if data_senales['action'][obs] == 1:
            plt.scatter(data_senales.index[obs], data_senales.close[obs], s=100, color='g')
        elif data_senales['action'][obs] == -1:
            plt.scatter(data_senales.index[obs], data_senales.close[obs], s=100, color='r')
        
    for obs in indice_max_data:
        plt.scatter(data.index[obs], data.close[obs], s=25, marker='s', color='b')
        
    for obs in indice_min_data:
        plt.scatter(data.index[obs], data.close[obs], s=25, marker='s',color='b')

        
    plt.show()
    
    
    
    
    
    

    
    
    
    
    
    
    