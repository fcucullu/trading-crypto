# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/farduh/xcapit/portfolios/src")
from PriceReader import PriceReader
import numpy as np
pr=PriceReader()
data=pr.get_price("BTCUSDT","15m").dropna().asfreq("15t").dropna()
data=pr.get_price("BTCUSDT","2h").dropna().asfreq("2h").dropna()
import matplotlib.pyplot  as plt
n=6
punto_extremo="maximos"
col="close"

from scipy.optimize import curve_fit

def recta(x,a,b):
    return (a*x+b)

indices = []
data["pendientemax"]=0
data["pendientemin"]=0
for i in range(len(data)):
    if (data[col].iloc[i] < data[col].iloc[i-n:i]).sum() :
        continue
    else:
        if i > n:
            if bool(indices) and (i-indices[-1])<= n:
                indices[-1]=i
            else :
                indices.append(i)
            
                
            if len(indices)>4 :#and (indices[-1]-indices[-2])>(2*n):
                X=indices[-4:]
                y=data["close"][indices[-4:]].values
                
                popt,dummy=curve_fit(recta,X,y)
                data["pendientemax"].iloc[i]=popt[0]
indices = []
         
for i in range(len(data)):
    if (data[col].iloc[i] > data[col].iloc[i-n:i]).sum() :
        continue
    else:
        if i > n:
            if bool(indices) and (i-indices[-1])<= n:
                indices[-1]=i
            else :
                indices.append(i)
            
                
            if len(indices)>4 :#and (indices[-1]-indices[-2])>(2*n):
                X=indices[-4:]
                y=data["close"][indices[-4:]].values
                
                popt,dummy=curve_fit(recta,X,y)
                data["pendientemin"].iloc[i]=popt[0]
               
               
data["pendientemax"]=data["pendientemax"].replace(0,np.nan).ffill()
data["pendientemin"]=data["pendientemin"].replace(0,np.nan).ffill()

plt.scatter(data["pendientemin"],
            data["close"].pct_change().shift(-1))

result=[]
for umbral in np.arange(0.1,10,0.1):
    result.append(data[abs(data["pendientemax"]-data["pendientemin"])<umbral]["returns"].corr(data[abs(data["pendientemax"]-data["pendientemin"])<umbral]["pendientemin"]))


data_temp=data[abs(data["pendientemax"]-data["pendientemin"])<0.1]
plt.scatter(data_temp["pendientemin"],data_temp["returns"],c=data_temp["pendientemax"])
plt.colorbar()

data["returns"]=data["close"].pct_change().shift(-1)

(data["pendientemax"]-data["pendientemin"])[abs(data["pendientemax"]-data["pendientemin"])<0.1]

import pandas as pd
pd.options.display.max_rows=200

(9500+100*data.reset_index().iloc[-600:]["pendiente"]).plot()
pp=data["close"].reset_index().iloc[indices]

pp[pp["index"]>data.index[-600]]["close"].plot(c="r",marker="o",linestyle="")

data.reset_index().iloc[10820:10840]["close"].plot()
for i in data.reset_index().index[10820:10840]:
    if  (abs(data["pendientemin"].iloc[i]-data["pendientemax"].iloc[i])<0.04):
        plt.axvline(i, ls='-', alpha=0.1, color = 'pink', lw = 5)   


plt.scatter(data["pendiente"],data["close"].pct_change().shift(-1))


import quantstats as qs
