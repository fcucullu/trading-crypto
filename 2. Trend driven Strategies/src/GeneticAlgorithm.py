#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
"""


import random
import pandas as pd

class GeneticAlgorithm:
    def __init__(self,
                 target_function,
                 parameter_bounds,
                 n_population=20,
                 pressure=4  ,
                 mutation_chance=0.3,
                 random_init_mult=1,
                 n_iter=30,**kwargs):
        """
        Se inicializa algoritmo genético
        
        * target_function: 
            clase dos métodos necesarios.
            1. function_to_optimize(self,**kwargs): funcion a la cual se la va a maximizar
        * parameter_bounds: dict
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        >>> parameter_bounds = {'x': (2, 4), 'y': (-3, 3)}
        * n_population: int
            La cantidad de individuos que habra en la poblacion
        * pressure: int
            Cuantos individuos se seleccionan para reproduccion. Necesariamente mayor que 2
        * mutation_chance: float
            La probabilidad de que un individuo mute. Valor entre 0 y 1        
        * random_init_mult:
            Multiplica el número de población aleatoria inicial para tener un muestreo más grande a partir del cual empezar
        * n_iter: int
            Número de iteraciones
        """
        
        self.target_function=target_function
        self.length = len(parameter_bounds) #La longitud del material genetico de cada individuo
        self.n_population = n_population 
        self.pressure = pressure #
        self.mutation_chance = mutation_chance #
        self.pbounds=parameter_bounds
        self.mult=random_init_mult
        self.n_iter=n_iter
        self.variables_name=list(parameter_bounds.keys())
        self.logfile=[]
        

    def individual(self):
        """
            Crea un individual
        """
        return {key:random.uniform(values[0],values[1]) for (key,values) in self.pbounds.items()}

  
    def crearPoblacion(self,num):
        """
            Crea una poblacion nueva de individuos
        """
        return [self.individual() for i in range(num)]

  

    def selection_and_reproduction(self,population):
        """
        Puntua todos los elementos de la poblacion (population) y se queda con los mejores
        guardandolos dentro de 'selected'.
        Despues mezcla el material genetico de los elegidos para crear nuevos individuos y
        llenar la poblacion (guardando tambien una copia de los individuos seleccionados sin
        modificar).
  
        Por ultimo muta a los individuos.
  
        """
        puntuados = [ (self.target_function.function_to_optimize(**i), i) for i in (population)] #Calcula el fitness de cada individuo, y lo guarda en pares ordenados de la forma (5 , [1,2,1,1,4,1,8,9,4,1])
        puntuados.sort(key=lambda x: x[0])
        population = [i[1] for i in puntuados] #Ordena los pares ordenados y se queda solo con el array de valores
       # population2 = pd.DataFrame(population).drop_duplicates().values.tolist()
          
        isel=(self.n_population-self.pressure)
        selected =  population[isel:] #Esta linea selecciona los 'n' individuos del final, donde n viene dado por 'pressure'
        
        #Se mezcla el material genetico para crear nuevos individuos
        for i in range(0,isel):
            for variable in self.variables_name:
                padre = random.sample(selected,1) #Se eligen dos padres
                population[i][variable] = padre[0][variable] #Se mezcla el material genetico de los padres en cada nuevo individuo

        return population,puntuados #El array 'population' tiene ahora una nueva poblacion de individuos, que se devuelven
      
    def mutation(self,population):
        """
            Se mutan los individuos al azar. Sin la mutacion de nuevos genes nunca podria
            alcanzarse la solucion.
        """
        for i in range(self.n_population-self.pressure):
            if random.random() <= self.mutation_chance: #Cada individuo de la poblacion (menos los padres) tienen una probabilidad de mutar
                variable = random.choice(self.variables_name) #Se elige un punto al azar
                
                nuevo_valor = random.uniform(self.pbounds[variable][0],self.pbounds[variable][1]) #y un nuevo valor para este punto
      
                #Es importante mirar que el nuevo valor no sea igual al viejo
                while nuevo_valor == population[i][variable]:
                    nuevo_valor = random.uniform(self.pbounds[variable][0],self.pbounds[variable][1])
      
                #Se aplica la mutacion
                population[i][variable] = nuevo_valor
      
        return population
      
  
    def run(self):
        
        results=[]
        
        population = self.crearPoblacion(self.mult*self.n_population)#Inicializar una poblacion
        #print("Poblacion Inicial:\n%s"%(population)) #Se muestra la poblacion inicial
        puntuados = [ (self.target_function.function_to_optimize(**i), i) for i in (population)] #Calcula el fitness de cada individuo, y lo guarda en pares ordenados de la forma (5 , [1,2,1,1,4,1,8,9,4,1])
        
        puntuados.sort(key=lambda x: x[0],reverse=True)

        population = [i[1] for i in puntuados] #Ordena los pares ordenados y se queda solo con el array de valores
        

        population = population[:self.n_population]
    
        #Se evoluciona la poblacion
        results=[]
        for i in (range(self.n_iter)):
    
            population,puntuados_anterior = self.selection_and_reproduction(population)
            population = self.mutation(population)
            self.dict_to_save(puntuados_anterior,results)
            
           # results.append({"baseline":puntuados_anterior[0]}.update(puntuados_anterior[1]))
            #print("\nPoblacion Final y Fitness") #Se muestra la poblacion evolucionada
        puntuados = [ (self.target_function.function_to_optimize(**i), i) for i in (population)]
        self.dict_to_save(puntuados,results)
        
        
        return pd.DataFrame(results).drop_duplicates()

    def dict_to_save(self,puntuados,results):
        for i in puntuados:
            output={"baseline":i[0]}
            output.update(i[1])
            results.append(output)
        
    

    #Se mezcla el material genetico para crear nuevos individuos
