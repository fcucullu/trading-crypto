#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:38:45 2020

@author: fcucullu
"""

import sys
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Optimizador')
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Estrategias explosivas + TSL')


import pandas as pd
import numpy as np
import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import estrategia_volumen_optimizador

from src.HyperParametersGenerator import BayesianHyperParametersGenerator
from src.HyperParametersGenerator import GeneticHyperParametersGenerator
from src.HyperParametersGenerator import RandomHyperParametersGenerator

from src.ModelValidation import LinearModelValidation
from src.ModelValidation import WalkForwardModelValidation
from src.ModelOptimization import ModelOptimization

path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia_explosiva\src\BTCUSDT_2h.csv'
data = estrategia_volumen_optimizador._get_price(path).dropna()


ob_func = estrategia_volumen_optimizador.Objective_Function(data, data.index[0], data.index[-1])

m=ModelOptimization(ob_func,BayesianHyperParametersGenerator,WalkForwardModelValidation)
#m=ModelOptimization(ob_func,RandomHyperParametersGenerator,WalkForwardModelValidation)

resultados = m.run_full_optimization({'obs_acumuladas': (1, 8+1), 
                             'ventana_objetivo': (1, 4+1),
                             'media': (1, 200+1),
                             'tipo_media': (1,2),
                             'tipo_retorno': (1,2), 
                             'periodo_bandas': (20, 30+1) ,
                             'desviacion_bandas': (2, 3.5)
                            },
                            0,   
                            test_size=0, #ES la cantidad de observaciones para el test
                            validation_size=0, #ES la cantidad de observaciones para la validacion
                            n_split=1, #Cantidad de splits para la validacion
                            n_iter=200,
                            #n_samples=300,
                            verbose=2 #Es para que me de los resultados en vivo
                            )

 
#resultados = pd.DataFrame()
#for tipo_media in ['EMA','MM']:
#    for tipo_retorno in ['NOABS']:#['ABS', 'NOABS']:
#        for media in range(1,200+1):
#            for obs_acumuladas in range(1,8+1):
#                for ventana_objetivo in range(1,4+1):
#                    for periodos in range(20,30):
#                        for desvios in [2, 2.5, 3, 3.5]:
#                            resultados = resultados.append( retorno_medio_segun_exceso_volumen_acumulado(df, obs_acumuladas, ventana_objetivo, media, tipo_media, tipo_retorno, periodos, desvios))




#result[1].to_csv("df_baseline_ATR_bayes_wf.csv")
#result[2].to_csv("df_validation_ATR_bayes_wf.csv")