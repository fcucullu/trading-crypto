#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase esquelo para definir la funciones a optimizar

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Recomendador import Recomendador
from .ObjectiveFunctionBase import AbstractObjectiveFunction
import quantstats as qs


#Funcion que necesita la clase Optimizador
class ObjectiveFunction(AbstractObjectiveFunction):    
    def __init__(self,time_serie,
                 date_init,
                 date_final,
                 output="sharpe",
                 periods=365*12,
                 plot=False):
        self.time_serie=time_serie
        self.date_init=date_init
        self.date_final=date_final
        self.output=output
        self.plot=plot
        self.maximum_volatility=100
        self.periods=periods
    
    def set_maximum_volatility(self,vol=100):
        self.maximum_volatility=vol
    
    def set_output(self,output="sharpe"):
        self.output=output
        
    def set_plot(self,plot=False):
        self.plot=plot
        
    def init_df_history(self):
        self.df_history=pd.DataFrame(columns=["riesgo","lecturas","pendiente","factorcom","date_init","date_final","volatilidad","sharpe","retorno_medio","retorno_total"])
        
    def update_df_history(self,riesgo,lecturas,pendiente,factorcom,returns):

        sharpe=np.sqrt(self.periods)*returns.mean()/returns.std(ddof=1)
        tot_return=(1+returns).cumprod()[-1]
        return_mean=(self.periods)*returns.mean()
        volatility=np.sqrt(self.periods)*returns.std(ddof=1)

        win_avg=returns[returns>0].dropna().mean()
        loss_avg=abs(returns[returns<0].dropna().mean())

    
        win_loss_ratio=win_avg/loss_avg
        win_prob=0
        if bool(len(returns)):
            win_prob=len(returns[returns>0])/len(returns)
        lose_prob=1-win_prob
        
        kelly_criterion=((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio
        


        self.df_history=self.df_history.append({"riesgo":riesgo,
                                                "lecturas":lecturas,
                                                "pendiente":pendiente,
                                                "factorcom":factorcom,
                                                "sharpe":sharpe,
                                                "volatilidad":volatility,
                                                "retorno_total":tot_return,
                                                "retorno_medio":return_mean,
                                                "criterio_kelly":kelly_criterion,
                                                "prob_ganar":win_prob,
                                                "retorno_medio_pos":win_avg,
                                                "retorno_medio_neg":loss_avg,
                                                "date_init":self.date_init,
                                                "date_final":self.date_final},ignore_index=True)
                        
    
    
    
    def get_returns(self,**kwargs):
    
         
        riesgo=kwargs["riesgo"]
        lecturas=kwargs["lecturas"]
        pendiente=kwargs["pendiente"]
        factorcom=kwargs["factorcom"]
    
        lecturas=int(lecturas)
        
        returns=self.time_serie.pct_change()
        
        recos=Recomendador(riesgo,lecturas,"2h","USDT",pendiente,factorcom)
        
        weights=recos.generar_recomendacion(prices=self.time_serie,
                                         ultima=False,
                                         start_date=self.date_init.strftime("%Y-%m-%d"),
                                         end_date=self.date_final.strftime("%Y-%m-%d")
                                         )
        
        
        weights=weights.loc[self.date_init:self.date_final]
        
        final_weights=(weights.shift(1)*(1+returns)).dropna() # muevo los pesos según los retornos
        
        final_weights=final_weights.apply(lambda x: x/final_weights.sum(axis=1)) # normalizo
        
        diff_weights=weights-final_weights # diferencia entre recomendación actual y nueva
        
        diff_weights = diff_weights.mul(np.where(diff_weights["USDT"]>=0,1,-1),axis=0)# lo que necesito comprar o vender de USDT siempre es positivo
        
        comision_values = pd.DataFrame(0.002,index=diff_weights.index,columns=diff_weights.columns)#df con valores de comisiones
        
        comision_values["USDT"]=0.001 # comision USDT

        comision=(comision_values*diff_weights[diff_weights>0]).replace(np.nan,0).sum(axis=1) # comisión total

        returns=((weights.shift(1)*returns).dropna().sum(axis=1)-comision)
        
        return returns
    
    def function_to_optimize(self,**kwargs): ##método necesario de la función para la optimización
            
        returns=self.get_returns(**kwargs)
        
        volatility=np.sqrt(self.periods)*returns.std(ddof=1)
        expected_return=((1+returns).cumprod()[-1])**(self.periods/len(returns))-1
        
        if volatility>self.maximum_volatility:
            return -100
        
        if self.output=="sharpe":
            result=expected_return/volatility
        elif self.output=="expected_return":
            result=expected_return
        elif self.output=="return_mean":
            result=(self.periods)*returns.mean()
        elif self.output=="volatility":
            result=volatility
        elif self.output=="kelly":
            result=qs.stats.kelly_criterion(np.log(1+returns).rolling(7).mean())
#        if self.plot==True:
#            df=pd.DataFrame(index=self.time_serie.loc[self.date_init:self.date_final].index)
#            df["returns"]=returns
#            (1+df.returns).cumprod().plot(label="portfolio")
#            #for icurr in (self.time_serie.columns):
#                #((1+self.time_serie[str(icurr)].pct_change()).loc[self.date_init:self.date_final].cumprod()).plot(label=icurr)
#            plt.legend()
#            
        #self.update_df_history(riesgo,lecturas,pendiente,factorcom,returns)
        
        return result
    

