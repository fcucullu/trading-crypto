"""
more info:
    https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
    https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9
    https://towardsdatascience.com/validation-methods-for-trading-strategy-development-1efea8284b02
    
"""
from .HyperParametersGenerator import AbstractHyperParametersGenerator
from .HyperParametersGenerator import BayesianHyperParametersGenerator
from .ModelValidation import LinearModelValidation
from .ModelValidation import WalkForwardModelValidation
from .XcapitRatioCalculator import XcapitRatioCalculator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class ModelOptimization:
    def __init__(self,
                 target_function,
                 generator:AbstractHyperParametersGenerator):
        """
        * target_function: *class*
            clase dos métodos necesarios.
                1. function_to_optimize(self,**kwargs): funcion a la cual se la va a maximizar
                2. set_dates(self,fecha_inicial,fecha_final) : método para fijar fecha desde donde hasta donde se obtiene el resultado a optimizar
        * generator: *class*
            clase con la que se generaran los modelos
        * validation: *class*
            clase con la cual se validara el modelo
        
        """
        
        self.target_function=target_function
        self.generator=generator(target_function)
        self.validation=WalkForwardModelValidation(target_function,self.generator)
    
        
    def run_full_optimization(self,
                              parameter_bounds,
                              train_size=200,
                              validation_size=0,
                              test_size=0,
                              n_split=1,
                              benchmark_model=None,
                              Xcapit_ratio_span=7,
                              **kwargs):
        """
        Se realiza una optimización total del modelo
        
        * parameter_bounds: *dict*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
        * train_size: int,
            int cantidad de datos que se utilizan para entrenar
            
        * test_size: int  
            Tamaño de la serie con la cual se testea el modelo
            Si el valor es 0 no se considera un muestra test
        * n_split:  int
            Número de regiones de validación utilizadas para realizar un walk-forward
            Si el valor es 0 no se realizará ninguna validación.
    
        Esta función retorna una tupla con:
            * Un pandas.DataFrame con los parámetros resultantes del entrenamiento, el resultado de entrenamiento, el resultado de test y el Xcapit_ratio
            * Un pandas.DataFrame con los resultados de las validaciones
        
        """
        train_sample,test_sample,t_v_samples=self.split_samples_in_train_val_test_regions(train_size,test_size,validation_size,n_split)
        validation_results=pd.DataFrame()
        #Entreno el modelo en la región de entrenamiento
        parameters,train_result=self.train_model(parameter_bounds,train_sample,**kwargs)
        result_train_test=parameters
        result_train_test.update({"train_result":train_result})
        if not t_v_samples.empty: # verifico que haya regiones de validation
            validation_results=self.validate_model(parameter_bounds,t_v_samples,n_split*validation_size,n_split,**kwargs)
            #Calculo el Xcapit ratio
            Xcapit_ratio,validation_results=self.get_xcapit_ratio(validation_results,benchmark_model,span=Xcapit_ratio_span)
            result_train_test.update({"Xcapit_ratio":Xcapit_ratio})
        if not test_sample.empty: 
            test_result=self.test_model(test_sample,parameters)
            result_train_test.update({"test_result":test_result})
        result_train_test=pd.DataFrame([result_train_test])
        
        return result_train_test,validation_results
        
    def split_samples_in_train_val_test_regions(self,train_size,test_size,validation_size,n_split):
        train_sample=pd.DataFrame()
        test_sample=pd.DataFrame()
        t_v_samples=pd.DataFrame()
        n_lecturas=len(self.target_function.time_serie.loc[self.target_function.date_init:self.target_function.date_final])
        min_n_lecturas=train_size+n_split*validation_size+test_size
        if n_lecturas<=min_n_lecturas:
            print(f"Se necesitan {train_size+(n_split+1)*test_size} lecturas para realizar la optimización y el tamaño de la muestra configurados por las fechas es de {len(self.target_function.time_serie.loc[self.target_function.date_init:self.target_function.date_final])}")
            exit(0)
        elif( type(train_size)==float) and ((train_size>1) or (train_size<=0)):
            print("train_size must be integer or float in (0,1)")
            exit(0)
        elif type(test_size)==float and (test_size>1 or test_size<0):
            print("test_size must be integer or float in (0,1)")
            exit(0)
        elif type(validation_size)==float and (n_split*validation_size>1 or validation_size<=0):
            print("validation_size must be integer or float in (0,1)")
            exit(0)
        
        temp=self.target_function.time_serie.loc[self.target_function.date_init:self.target_function.date_final]
        #Defino muestra test
        if test_size!=0:
            temp,test_sample=train_test_split(temp,test_size=test_size,shuffle=False)        
        #Defino muestra train
        if train_size!=1:
            train_sample=train_test_split(temp,test_size=train_size,shuffle=False)[1]
            if n_split!=0 and validation_size!=0:
                t_v_samples =train_test_split(train_sample,test_size=(train_size+n_split*validation_size),shuffle=False)[1]
        else:
            train_sample=temp
            t_v_samples=temp
        return train_sample,test_sample,t_v_samples
        
    def get_xcapit_ratio(self,validation_results,benchmark_price,span=2.5):
        
        return XcapitRatioCalculator().get_xcapit_ratio(validation_results,benchmark_price,span)
        
        
    def generate_samples(self, parameter_bounds,initial_date,final_date,**kwargs):
        """
        Genera muestras con distintos hiperparametros
        * parameter_bounds: *dictionario*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
        * initial_date: *datetime*
            fecha en la que empieza ejecutarse el modelo
        * final_date: *datetime*
            fecha en la que termina ejecutarse el modelo
        * **kwargs:
            hiperparametros de los modelos generadores
        """
        self.target_function.set_dates(initial_date,final_date)
        return self.generator.generate_parameters(parameter_bounds,**kwargs)
    
    def train_model(self,parameter_bounds,train_sample,**kwargs):
        """
        Entrena el modelo
        inputs:
            * parameter_bounds: *dictionario*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
            * train_sample
            * **kwargs :
                parámetros de la maximización
        
        """
        initial_date=train_sample.index[0]
        final_date=train_sample.index[-1]
        df=self.generate_samples(parameter_bounds,initial_date,final_date,**kwargs)
        return df.drop(columns="baseline").loc[df["baseline"].idxmax()].to_dict(),df["baseline"].loc[df["baseline"].idxmax()]
        
    
    def validate_model(self,parameter_bounds,t_v_samples,validation_size,n_split,**kwargs):
        """
        Obtiene	los resultado de las regiones de validación
        * parameter_bounds: *dictionario*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        * t_v_samples:
            muestra que se utilizara para realizar la validación
        
        * validation_size: *float, int*
            Tamaño de la serie con la cual se validará el modelo. Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en la validación. Si es entero, representa el número absoluto de muestras de validación.  
            
        * **kargs:
            argumentos específicos del modelo
            
        """
        
        initial_date=t_v_samples.index[0]
        final_date=t_v_samples.index[-1]
        
        
        return self.validation.validate_models(parameter_bounds,initial_date,final_date,validation_size,n_split=n_split,**kwargs)
    
    def test_model(self,test_sample,parameters):
        """
        Calcula los resultados de la función a optimizar entre las fechas que tiene de entrada para los hiperpárametros en las filas de df_samples

        
        *test_sample: muestra que se realizará para entrenar el modelo
        
        * parameters: parámetros a evaluar en test
        """
        initial_date=test_sample.index[0]
        final_date=test_sample.index[-1]
        
        self.target_function.set_dates(initial_date,final_date)
        
        test_result=self.target_function.function_to_optimize(**parameters)    
            
        return test_result
        
    
    @classmethod
    def defaut_bayesian_linear(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        return m
    
    @classmethod
    def defaut_bayesian_walkforward(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        return m

    @classmethod
    def run_defaut_bayesian_linear(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        df=m.run_full_optimization()
        return df
    @classmethod
    def run_defaut_bayesian_walkforward(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),WalkForwardModelValidation(f))
        df=m.run_full_optimization()
        return df
     
        

