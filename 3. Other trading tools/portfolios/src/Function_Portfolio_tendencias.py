# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/farduh/xcapit/optimizador/src")
from Recomendador import Recomendador
from ObjectiveFunctionBase import AbstractObjectiveFunction



#Funcion que necesita la clase Optimizador
class ObjectiveFunction(AbstractObjectiveFunction):    
    def __init__(self,time_serie,date_init,date_final,output="sharpe",plot=False):
        self.time_serie=time_serie
        self.date_init=date_init
        self.date_final=date_final
        self.output=output
        self.plot=plot
        self.maximum_volatility=100
        
    
    def set_market_status(self,market_status):
        self.market_status=market_status
    
    def set_maximum_volatility(self,vol=100):
        self.maximum_volatility=vol
    
    def set_output(self,output="sharpe"):
        self.output=output
     
    def get_returns(self,**kwargs):
        
        riesgo_up=kwargs["riesgo_up"]
        lecturas_up=kwargs["lecturas_up"]
        pendiente_up=kwargs["pendiente_up"]
        factorcom_up=4.38#kwargs["factorcom_up"]
        riesgo_down=kwargs["riesgo_down"]
        lecturas_down=kwargs["lecturas_down"]
        pendiente_down=kwargs["pendiente_down"]
        factorcom_down=4.38#kwargs["factorcom_down"]

        lecturas_up=int(lecturas_up)
        lecturas_down=int(lecturas_down)
        
    
        recosup=Recomendador(riesgo_up,lecturas_up,"2h","USDT",pendiente_up,factorcom_up)
        recosdown=Recomendador(riesgo_down,lecturas_down,"2h","USDT",pendiente_down,factorcom_down)

        recoup=recosup.generar_recomendacion(prices=self.time_serie,
                                             ultima=False,
                                             start_date=self.date_init.strftime("%Y-%m-%d"),
                                             end_date=self.date_final.strftime("%Y-%m-%d"))
        recodown=recosdown.generar_recomendacion(prices=self.time_serie,
                                                 ultima=False,
                                                 start_date=self.date_init.strftime("%Y-%m-%d"),
                                                 end_date=self.date_final.strftime("%Y-%m-%d")
                                            )

       
        recodown=recodown.asfreq("2h").ffill().loc[self.date_init:self.date_final]
        recoup=recoup.asfreq("2h").ffill().loc[self.date_init:self.date_final]
        market_status=self.market_status.loc[self.date_init:self.date_final]
        reco=recoup[market_status.action==1].append(recodown[market_status.action==-1])
        reco=reco.sort_index()

        comision=reco.diff()[reco.diff()>0].sum(axis=1)*0.001
        
        returns=(((reco.shift(1)*self.time_serie.pct_change()).dropna().sum(axis=1)*(1-comision)))
        
        return returns
    
    
    def get_volatility(self,returns):
        return np.sqrt(365*12)*returns.std(ddof=1)

    def get_mean_return(self,returns):
        return (365*12)*returns.mean()

    def get_cumulative_return(self,returns):
        return (1+returns).cumprod()[-1]

    def get_sharpe(self,returns):
        return np.sqrt(365*12)*returns.mean()/returns.std(ddof=1)
    
    def function_to_optimize(self,**kwargs): ##método necesario de la función para la optimización
            
        returns = self.get_returns(**kwargs)
        
        volatility=self.get_volatility(returns)        
        if volatility>self.maximum_volatility:
            return -100
        
        result=self.get_sharpe(returns)
        
        
        return result
    
