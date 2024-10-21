from abc import ABC,abstractclassmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class AbstractModelValidation(ABC):
    def __init__(self,target_function,generator):
        self.target_function=target_function
        self.generator=generator
        super().__init__()
    @abstractclassmethod
    def validate_models():
        pass
    
class LinearModelValidation(AbstractModelValidation):
    
    
    def validate_models(self,parameter_bounds,date_init,date_final,validation_size,**kwargs):
        """
        Se aplica lineal validation sobre las muestras de entrenamiento
        
        * df_samples:
            dataframe con los hiperparámetros que van a ser validados con este método
        
        * validation_size: *float, int or None, optional (default=None)*
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
        """
        dates=pd.DatetimeIndex(self.target_function.time_serie.loc[date_init:date_final].index)
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        
        dates_train={"inicial":index_train[0],
                     "final":index_train[-1]}
    
        dates_validation={"inicial":index_validation[0],
                    "final":index_validation[-1]}
            
        
        df_results=pd.DataFrame()
        
        self.target_function.set_dates(dates_train["inicial"],dates_train["final"])    
            
        df=self.generator.generate_parameters( parameter_bounds,**kwargs)
            
        idxmax=df.idxmax()
            
        result=df.loc[idxmax]
        result=result.rename(index={"baseline":"train"})
        
        self.target_function.set_dates(dates_validation["inicial"],dates_validation["final"])
        validation_result=self.target_function.function_to_optimize(**(result.to_dict()))
        result["validation"]=validation_result
        result["val_date_init"]=dates_validation["inicial"]
        result["val_date_final"]=dates_validation["final"]
        result["train_date_init"]=dates_train["inicial"]
        result["train_date_final"]=dates_train["final"]
    
        df_results.append(result,ignore_index=True)
        
        return df_results
            
class WalkForwardModelValidation(AbstractModelValidation):
    
    
    
    def validate_models(self,parameter_bounds,date_init,date_final,validation_size,n_split,anclaje=False,**kwargs):
        """
        Se aplica walkforward sobre las muestras de entrenamiento
        * parameter_bounds:
            límites de los parámetros del modelo
    
        * date_init: datetime
            fecha en la que comienza la muestra
            
        * date_final: datetime
            fecha en la que termina la muestra
            
        * anclaje: *False or True (default=False)*
            True en caso que se quiera hacer la validación con anclaje.
        
        * validation_size: *float, int or None, optional (default=None)*
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
        """
        dates=pd.DatetimeIndex(self.target_function.time_serie.loc[date_init:date_final].index)
        
        
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        index_validation_array=np.array_split(index_validation,n_split)
        
        dates_train=[]
        dates_validation=[]
        train_shift=0
        
        for i in range(0,n_split):
            index_validation=index_validation_array[i]
            
            
            dates_train.append({"inicial":index_train[train_shift],
                                "final":dates[len(index_train)+train_shift]})
    
            dates_validation.append({"inicial":index_validation[0],
                                "final":index_validation[-1]})
            if anclaje==False:
                train_shift+=len(index_validation)
            
        
        
        df_results=pd.DataFrame()
        for n_iter in range(0,n_split):
            
            self.target_function.set_dates(dates_train[n_iter]["inicial"],dates_train[n_iter]["final"])    
            
            df=self.generator.generate_parameters( parameter_bounds,**kwargs)
            
            idxmax=df["baseline"].idxmax()
            
            result=df.loc[idxmax]
            
            result=result.rename(index={"baseline":"train"})
            
            self.target_function.set_dates(dates_validation[n_iter]["inicial"],dates_validation[n_iter]["final"])
            
            validation_result=self.target_function.function_to_optimize(**(result.to_dict()))
            result["validation"]=validation_result
            result["n_iter"]=n_iter
            result["val_date_init"]=dates_validation[n_iter]["inicial"]
            result["val_date_final"]=dates_validation[n_iter]["final"]
            result["train_date_init"]=dates_train[n_iter]["inicial"]
            result["train_date_final"]=dates_train[n_iter]["final"]
        
            df_results=df_results.append(result,ignore_index=True)
        
        df_results.set_index("n_iter")
        
        
        
        return df_results
        
    
        
