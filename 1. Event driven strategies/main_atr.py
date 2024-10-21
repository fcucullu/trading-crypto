#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:38:45 2020

@author: fcucullu
"""

import sys
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Optimizador')
sys.path.append(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia_explosiva\src')


path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\pivot_points\src'
sys.path.append(path)
from funciones_auxiliares import get_puntos_extremos


import warnings
warnings.filterwarnings("ignore")

import estrategia_atr_optimizador

from src.HyperParametersGenerator import BayesianHyperParametersGenerator
from src.HyperParametersGenerator import GeneticHyperParametersGenerator
from src.HyperParametersGenerator import RandomHyperParametersGenerator

from src.ModelValidation import LinearModelValidation
from src.ModelValidation import WalkForwardModelValidation
from src.ModelOptimization import ModelOptimization

path = r'C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia_explosiva\src\BTCUSDT_2h.csv'
data = estrategia_atr_optimizador._get_price(path).dropna()

#Identifico PP para calcular maximos DD
indice_max_data = get_puntos_extremos(data, 'close', 10, 'maximos')
indice_min_data = get_puntos_extremos(data, 'close', 10, 'minimos')


ob_func = estrategia_atr_optimizador.Objective_Function(data, data.index[0], data.index[-1])
ob_func.definir_pp(indice_max_data, indice_min_data)


m=ModelOptimization(ob_func,BayesianHyperParametersGenerator,WalkForwardModelValidation)
#m=ModelOptimization(ob_func,RandomHyperParametersGenerator,WalkForwardModelValidation)



resultados = m.run_full_optimization({'mult_high': (1, 4+1), 
                             'mult_low': (1, 4+1),
                             'chand_window': (1, 50+1),
                             'media': (1,30+1),    
                            },
                            -10, #umbral a superar   
                            test_size=0, #ES la cantidad de observaciones para el test
                            validation_size=0, #ES la cantidad de observaciones para la validacion
                            n_split=1, #Cantidad de splits para la validacion
                            n_iter=200,
                            #n_samples=300,
                            verbose=2, #Es para que me de los resultados en vivo
                            )



#result[1].to_csv("df_baseline_ATR_bayes_wf.csv")
#result[2].to_csv("df_validation_ATR_bayes_wf.csv")