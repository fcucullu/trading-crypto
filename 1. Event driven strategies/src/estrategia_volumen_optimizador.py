import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia_explosiva\src')
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Optimizador\src')

from funciones_volumen import *
from ta import volatility
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

    
class Objective_Function(AbstractObjectiveFunction):
        
    def retorno_medio_segun_exceso_volumen_acumulado(self,
                                                     obs_acumuladas, 
                                                     ventana_objetivo, 
                                                     media,
                                                     tipo_media='EMA',
                                                     tipo_retorno='ABS', 
                                                     periodo_bandas=30, 
                                                     desviacion_bandas=1):
        '''
        La funcion detecta cuando el volumen de las obs_acumuladas sobrepasa una 
        media movil y calcula el retorno acumulado que se obtendria en la vengana 
        objetivo. La ventana objetivo se define dede el OPEN del dia siguiente a
        la senial hasta el CLOSE del final de la ventana.
        Ejemplo:
            Ventana_objetivo = 2 >>> Retorno = Close[2+1] / Open[1] - 1
            Ventana_objetivo = 0 >>> Retorno = Close[0+1] / Open[1] - 1
    
        La direccion de la operacion se determina por la banda de bollinger
        que rompe el precio.
    
        inputs:
            df = Data financiera con el dato mas reciente al final
            obs_acumuladas = observaciones hacia atras para acumular volumen
            ventana_objetivo = cantidad de dias hacia adelante a analizar
            media = periodos de la media movil
            tipo_media = tipo de media movil. Acepta 'EMA' y 'MM'
            tipo_retorno = tipo del retorno futuro calculado. Acepta 'ABS' y 'NOABS'.
                El primero es el retorno absoluto de la ventana objetivo y el segundo
                es el retorno direccional segun la direccion de las ultimas obs_acumuladas.
            periodo_bandas = es la cantidad de observaciones hacia atras para calcular las bandas.
            desviacion_bandas = es la cantidad de desvios estandar que el CLOSE debe superar 
                para calcular la direccion de la serie.
        
        output:
            Dataframe con las siguientes columnas
                'media': media analizada
                'tipo_media': tipo de media analizada
                'tipo_retorno': tipo de retorno analizado
                'obs_acumuladas: cantidad de observaciones en las que se acumulo el vol
                'ventana_objetivo': rango futuro en donde se maximiza retorno
                'trades': cantidad de seniales detectadas
                'amplitud_mediana': La amplitud mediana de las velas (high-low)
                'retorno_futuro': retorno medio futuro obtenido por la estrategia
                'perido_bandas': obs acumuladas para calcular bandas
                'desvio_bandas': cantidad de desvios para calcular direccion
                'ret_esperado': Es la metrica para definir que estrategia es mejor.
                        Es el retorno promedio por la probabilidad de operar.
                        La probablidad de trade esta calculada como 1/len(df)*365
        '''
        
        #Funcion que setea la direccion con Bandas de Bollinger
        def set_direccion_mercado(self,
                                  periodo_bandas=30,
                                  desviacion_bandas=1): #ndev = 1.75
            
            bb_high =  volatility.bollinger_hband(self.time_serie['close'], n=periodo_bandas, ndev=desviacion_bandas, fillna=False)
            bb_low =   volatility.bollinger_lband(self.time_serie['close'], n=periodo_bandas, ndev=desviacion_bandas, fillna=False)
            
            self.time_serie['direccion'] = 0
            
            self.time_serie['direccion'] = np.where(self.time_serie['close'] > bb_high,1,self.time_serie['direccion'])
            self.time_serie['direccion'] = np.where(self.time_serie['close'] < bb_low,-1,self.time_serie['direccion'])
            del bb_high,bb_low
            
            return 
        
        #Casteo DF vacia para guardar los resultados
        self.resultados = pd.DataFrame(columns=['media',
                                                'tipo_media',
                                                'tipo_retorno',
                                                'obs_acumuladas',
                                                'ventana_objetivo',
                                                'trades',
                                                'amplitud_mediana',
                                                'retorno_futuro',
                                                'periodo_bandas',
                                                'desvios_bandas',
                                                'ret_esperado'])
                
        #casteo una tabla interna para no toquetear la original
        tabla = self.time_serie.copy()   
        
        #Calculo el volumen acumulado
        tabla['volume_acum'] = tabla['volume'].rolling(media).sum()    
                 
        #Calculo los retornos futuros en base a la ventana objetivo
        if tipo_retorno == 'ABS':
            tabla['ret_futuro'] = abs(tabla['close'].shift(-ventana_objetivo-1)/tabla['open'].shift(-1) - 1 )
        elif tipo_retorno == 'NOABS':
            tabla = set_direccion_mercado(tabla, periodo_bandas, desviacion_bandas)
            tabla['ret_futuro'] = 0
            tabla['ret_futuro'] = np.where(tabla['direccion'] == 1,
                                  tabla['close'].shift(-ventana_objetivo-1)/tabla['open'].shift(-1) - 1,
                                  tabla['ret_futuro'])
            tabla['ret_futuro'] = np.where(tabla['direccion'] == -1,
                                  tabla['open'].shift(-1)/tabla['close'].shift(-ventana_objetivo-1) - 1,
                                  tabla['ret_futuro'])
        else:
            print('Ha seleccionado un tipo de retorno invalido. Elija entre: "ABS" o "NOABS"')
            return
        
        #Creo columna vacia para la senial
        tabla['signal'] = 0    
        #Calculo la amplitud de las velas en la misma ventana objetivo
        tabla['amplitud'] = abs(tabla['high'].shift(-ventana_objetivo-1) - tabla['low'].shift(-1))/tabla['low'].shift(-1)
        
        #Bifurcacion del analisis segun la media elegida: aritmetica o exponencial
        if tipo_media == 'MM':
            tabla['media_volumen'] = tabla['volume_acum'].rolling(media).mean()
        elif tipo_media == 'EMA':   
            tabla['media_volumen'] = tabla['volume_acum'].ewm(span=media, adjust=False).mean()       
        else:
            print('Ha seleccionado un tipo de media invalido. Elija entre: "EMA" o "MM"')
            return
        
        # Detecto los las observaciones en donde el volumen observado es mayor a la media
        if tipo_retorno == 'ABS':
            tabla['signal'] = (tabla['volume_acum'] > tabla['media_volumen']).astype('int')
        elif tipo_retorno == 'NOABS': 
            tabla['signal'] = ((tabla['volume_acum'] > tabla['media_volumen']) & (tabla['direccion'] != 0)).astype('int')
    
        #Calculo metricas de analisis
        amplitud = tabla.loc[tabla['signal'] == 1, 'amplitud'].median()
        ret_futuro = tabla.loc[tabla['signal'] == 1, 'ret_futuro'].mean()
        observaciones = len(tabla[tabla['signal'] == 1])
        ret_esperado = ret_futuro / len(self.time_serie) * 365 #Retorno esperado ajustado por probabilidad y anualizado
        
        self.resultados = self.resultados.append({'media': media,
                                      'tipo_media': tipo_media,
                                      'tipo_retorno': tipo_retorno,
                                      'obs_acumuladas': obs_acumuladas,
                                      'ventana_objetivo': ventana_objetivo,
                                      'trades': observaciones,
                                      'amplitud_mediana': amplitud,
                                      'retorno_futuro': ret_futuro,
                                      'periodo_bandas': periodo_bandas, 
                                      'desvios_bandas': desviacion_bandas,
                                      'ret_esperado': ret_esperado}, ignore_index=True)
            

        return
        
    def select_best_params_volumen(self):
        
        best = self.resultados[self.resultados['ret_esperado'] == self.resultados['ret_esperado'].max()]
        
        #Elijo el primero por que sera el de los parametros mas cortos
        best = best.iloc[0]
        
        return best['ret_esperado']

    def function_to_optimize(self,
                             obs_acumuladas, 
                             ventana_objetivo, 
                             media,
                             tipo_media,
                             tipo_retorno, 
                             periodo_bandas, 
                             desviacion_bandas):
        
        obs_acumuladas = int(obs_acumuladas)
        ventana_objetivo = int(ventana_objetivo)
        media = int(media)
        periodo_bandas = int(periodo_bandas)
        tipo_media = 'EMA' if tipo_media >= 1 else 'MM'
        tipo_retorno = 'ABS' if tipo_retorno >= 1 else 'NOABS'
                
        
        self.retorno_medio_segun_exceso_volumen_acumulado(
                                                 obs_acumuladas, 
                                                 ventana_objetivo, 
                                                 media,
                                                 tipo_media,
                                                 tipo_retorno, 
                                                 periodo_bandas, 
                                                 desviacion_bandas)
        try:
            best = self.select_best_params_volumen()
        except:
            best = -99
        
        return best


#dias = 12*7*38 #38 semanas
#path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Estrategias explosivas + TSL\BTCUSDT_2h.csv'
#df = _get_price(path)
#df = df.dropna(axis=0)
#df = df[-dias:]
#df = df.dropna(axis=0)
#
#    
## EXPERIMETNO!!
#resultados = pd.DataFrame()
#for tipo_media in ['EMA','MM']:
#    for tipo_retorno in ['NOABS']:#['ABS', 'NOABS']:
#        for media in range(1,200+1):
#            for obs_acumuladas in range(1,8+1):
#                for ventana_objetivo in range(1,4+1):
#                    for periodos in range(20,30):
#                        for desvios in [2, 2.5, 3, 3.5]:
#                            resultados = resultados.append( retorno_medio_segun_exceso_volumen_acumulado(df, obs_acumuladas, ventana_objetivo, media, tipo_media, tipo_retorno, periodos, desvios))
#        
#
#resultados.to_csv(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Estrategias explosivas + TSL\resultados_volumen.csv')
#
#



