from abc import ABC,abstractclassmethod
import random
import pandas as pd 
import numpy as np
import itertools
from bayes_opt import BayesianOptimization
from bayes_opt import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from .GeneticAlgorithm import GeneticAlgorithm



class AbstractHyperParametersGenerator(ABC):
    def __init__(self,target_function):
        self.target_function=target_function
        super().__init__()
        
    @abstractclassmethod
    def generate_parameters():
        """
        No
        """
        pass


class RandomHyperParametersGenerator(AbstractHyperParametersGenerator):
    
    def generate_parameters(self,parameter_bounds,n_samples=1000,verbose=0):
        """
        inputs:
            
        * parameter bounds: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        
        >>> parameter_bounds = {'x': (2, 4), 'y': (-3, 3)}
        * n_samples: 
            numero de samples aleatorias que se desea generar
        
        """
        
        if verbose!=0:
            print("",end=' | ')
            print("{x:<{s}}".format(x="iter",s=9),end=' | ')
            for key in parameter_bounds.keys(): 
                print("{x:<{s}}".format(x=key,s=9,p=4), end=' | ') 
            print("{x:<{s}}".format(x="baseline",s=9,p=4), end=' | ') 
            print("")        

        
        
        results=[]        
        for i in (range(n_samples)):
            result={}
            for key,values in parameter_bounds.items():
                result.update({key:random.uniform(values[0],values[1])})
            result.update({"baseline":self.target_function.function_to_optimize(**result)})
            
            if verbose!=0:
                print("",end=' | ')
                print("{x:<{s}}".format(x=i,s=9),end=' | ')
                for key in result.keys(): 
                    print("{x:<{s}.{p}}".format(x=result[key],s=9,p=4), end=' | ') 
                print("")        
    
            results.append(result)
        
        return pd.DataFrame(results)

class GridHyperParametersGenerator(AbstractHyperParametersGenerator):
    
    def generate_parameters(self,parameter_bounds,parameter_steps,verbose=0):
        """
        inputs:
            
        * parameter_bounds: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        
        >>>pbounds = {'x':  (2, 4), 'y': (-3, 3)}
        * parameter_steps: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        
        >>>steps = {'x':  0.2, 'y': 0.3}
        
        """
        
        
        
        iterators={}
        num_iter=1
        for key,values in parameter_bounds.items():
            iterator=np.arange(values[0],values[1]+1e-10,parameter_steps[key])
            iterators.update({key:iterator})
            num_iter*=len(iterator)
                             
        print("it requires {} iteration".format(num_iter))
        
        if verbose!=0:
            print("",end=' | ')
            print("{x:<{s}}".format(x="iter",s=9),end=' | ')
            for key in parameter_bounds.keys(): 
                print("{x:<{s}}".format(x=key,s=9,p=4), end=' | ') 
            print("{x:<{s}}".format(x="baseline",s=9,p=4), end=' | ') 
            print("")        

        
        results=[]
        for values in (itertools.product(*iterators.values())):
            
            result=dict(zip(iterators.keys(), values))
            result.update({"baseline":self.target_function.function_to_optimize(**result)})
            results.append(result)
            if verbose!=0:
                    print("",end=' | ')
                    print("{x:<{s}}".format(x=values,s=9),end=' | ')
                    for key in result.keys(): 
                        print("{x:<{s}.{p}}".format(x=result[key],s=9,p=4), end=' | ') 
                    print("")        
        
        return pd.DataFrame(results)
    
class BayesianHyperParametersGenerator(AbstractHyperParametersGenerator):
            
    def generate_parameters(self,
                            parameter_bounds,
                            n_iter=100,
                            init_points=5,
                            logs_path="logs/",
                            save_logfile=False,
                            load_logfile=False,
                            save_path="bayes_opt_logs.json",
                            load_paths=["bayes_opt_logs.json"],
                            verbose=0,**kwargs):
        """
        inputs:
            
        * parameter_bounds: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
        * n_iter: 
            numero de iteraciones del algoritmo
        * logs_path: 
            directorio donde se guardaran los logs
        * save_log: 
            save logfile
        * load_log: 
            load logfiles
        * save_path: 
            nombre que recibira el logfile al guardarse
        * load_paths: 
            lista con nombres de los logfiles a partir de los que continuará iterando el método
        * verbose:
            En caso de ser igualado a dos imprimirá una	lista de los puntos por	los que	va pasando
     
        """
        results=[]
        optimizer=BayesianOptimization(
                    f=self.target_function.function_to_optimize,
                    pbounds=parameter_bounds,
                    verbose=verbose)
        
        if load_logfile:
            load_logs(optimizer, logs=[logs_path+load_path for load_path in load_paths]);  
        if save_logfile:
            logger = JSONLogger(path=logs_path+save_path)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)        
        
        optimizer.maximize(n_iter=n_iter,init_points=init_points)
        
        for optimizer_result in optimizer.res:
            target=optimizer_result["target"]
            result=optimizer_result["params"]
            
            result.update({"baseline":target})
            
            results.append(result)
            
        return pd.DataFrame(results)

class GeneticHyperParametersGenerator(AbstractHyperParametersGenerator):
    
    def generate_parameters(self,parameter_bounds,**kwargs):
        """
        Se inicializa algoritmo genético
        
        * parameter_bounds: *dict*
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        >>> parameter_bounds = {'x': (2, 4), 'y': (-3, 3)}
        * n_population: *int*
            La cantidad de individuos que habra en la poblacion
        * pressure: *int*
            Cuantos individuos se seleccionan para reproduccion. Necesariamente mayor que 2
        * mutation_chance: *float*
            La probabilidad de que un individuo mute. Valor entre 0 y 1        
        * random_init_mult: *float*
            Multiplica el número de población aleatoria inicial para tener un muestreo más grande a partir del cual empezar
        * n_iter: *int*
            Número de iteraciones
        """
        alggen=GeneticAlgorithm(self.target_function,parameter_bounds,**kwargs)
        return alggen.run()
  
