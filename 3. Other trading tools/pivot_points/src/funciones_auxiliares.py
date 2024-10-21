import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

######################################################

def get_pivotpoints(serie_price, n):
    '''
    Deteta puntos pivote dentro del intervalo de 'n' observaciones.
    - serie_price: pandas.series
    - n: cantidad de obs del analisis
    '''
    maxs, mins, pps = [],[],[]
     
    for i in range(len(serie_price)):
        if (((serie_price[i] < serie_price[i-n:i]).unique()).any() == True) or (((serie_price[i] < serie_price[i+1:i+1+n]).unique()).any() == True) :
            pass
        else:
            if i > n:
                pps.append((i,serie_price[i]))
                maxs.append(i)
        if (((serie_price[i] > serie_price[i-n:i]).unique()).any() == True) or (((serie_price[i] > serie_price[i+1:i+1+n]).unique()).any() == True) :
            pass
        else:
            if i > n:
                pps.append((i,serie_price[i]))
                mins.append(i)
    return maxs,mins,pps

######################################################
    
def graph_price_plus_pivotpoints(serie_price, maxs, mins):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(serie_price, linewidth = 1, color = 'black')
    plt.legend('Price', prop={'size': 20})
    plt.title('Price + PivotPoints')
    plt.xlabel('Observations')
    plt.ylabel('Price')
    
    for obs in maxs:
        plt.scatter(serie_price.index[obs], serie_price[obs], s=50, color='g')
    for obs in mins:
        plt.scatter(serie_price.index[obs], serie_price[obs], s=50, color='r')
 
    plt.show()

######################################################

def identify_horizontal_lines(pps_list=list):
    pass

def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12






