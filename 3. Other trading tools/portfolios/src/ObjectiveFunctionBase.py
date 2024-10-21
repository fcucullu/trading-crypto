#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:42:38 2020

@author: farduh
"""
from abc import ABC,abstractclassmethod
from datetime import datetime
class AbstractObjectiveFunction(ABC):
    
    def __init__(self,time_serie,date_init,date_final,**kwargs):
        """
        * time_serie: *pandas.DataFrame*
            Contiene la serie temporal que se necesitará para usar en la función a optimizar
        * date_init: *datetime*
            Fecha a partir de la que empieza la estrategia de la función
        * date_final: *datetime*
            Fecha en la que termina el estrategia de la función
        * **kwargs: *dict*
             Contiene variables externas
             
        """
        self.time_serie=time_serie
        self.date_init=date_init
        self.date_final=date_final
        self.variable_externa=kwargs
    
    def set_dates(self,date_init,date_final):
        """
        Permite fijar fecha de inicio y de finalización
        * date_init: *datetime*
            Fecha a partir de la que empieza la estrategia de la función
        * date_final: *datetime*
            Fecha en la que termina el estrategia de la función
        
        
        """
        
        self.date_init=date_init
        self.date_final=date_final
    
    @abstractclassmethod
    def function_to_optimize():
        """
        Definir function a optimizar cuyos argumentos son sólo las variables a optimizar.
        Ejemplo como definir:
            
        >>> def function_to_optimize(self,a1,a2,a3):
            
        donde a1, a2 y a3 son las variables a optimizar.
        Se puede acceder al pandas.DataFrame con la serie temporal en la función de la siguiente forma:
        >>> df=self.time_serie
        
        Es importante que se defina el rango en el cual se esta valuando la función utilizando
        
        >>> df_result=df_result.loc[self.date_init,self.date_final]
        
        Para acceder a otra variable externa que se hayan definido la inicialización de la clase, se debe proceder de la siguiente manera (suponiendo como variables externas "vext1" y "vext2")
        
        >>>variable_externa_1=self.variable_externa["vext1"]
        >>>variable_externa_2=self.variable_externa["vext2"]
        
        """
        pass
    
    
    
    
    
    