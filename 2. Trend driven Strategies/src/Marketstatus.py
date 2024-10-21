# -*- coding: utf-8 -*-
import numpy as np

class LateralMarketIdentification():
        
    def __init__(self,prices_short,prices_long,freq_short="2h",freq_long="1d"):
        
        data_s=prices_short.copy()
        data_l=prices_long.copy()
        
        if data_s.index.freq==None : data_s=data_s.asfreq(freq_short)  
        if data_l.index.freq==None : data_l=data_l.asfreq(freq_long)
            
        
        data_l = self.get_pivots(data_l) 
        df = data_l.copy().asfreq(freq_short)
        df[data_s.columns] = data_s[data_s.columns]
        self.df=df.ffill()
        
        
        
    def identify_lateral(self,multiplicador):
        
        df=self.df
        df["cond"]=0
        #Condiciones sobre close
        df["cond"]+=(df['close'] < (df['pp'] + multiplicador*df['pp']/100)) 
        df["cond"]+=(df['close'] > (df['pp'] - multiplicador*df['pp']/100)) 
        #Condiciones sobre close anterior
        df["cond"]+=(df['close'].shift(1) < (df['pp'] + multiplicador*df['pp']/100)) 
        df["cond"]+=(df['close'].shift(1) > (df['pp'] - multiplicador*df['pp']/100)) 
        #Condiciones sobre open
        df["cond"]+=(df['open'] < (df['pp'] + multiplicador*df['pp']/100))
        df["cond"]+=(df['open'] > (df['pp'] - multiplicador*df['pp']/100))
        #Condiciones sobre open anterior
        df["cond"]+=(df['open'].shift(1) < (df['pp'] + multiplicador*df['pp']/100))
        df["cond"]+=(df['open'].shift(1) > (df['pp'] - multiplicador*df['pp']/100))
        
        df["lateral"] = np.where(df["cond"]==8,1,0)
        return df["lateral"]
    
        
    def get_pivots(self,data):
        '''
        Esta función devuelve los soportes y resistencias usando la data del periodo (vela) anterior. Por ejemplo, es
        útil para calcular los pivot points del día (si el dataframe es diario). 
        Los parámetros de entrada son:
        - data: Un dataframe con columnas: "close", "open", "high" y "low".
        '''
        
        data_temp = data.copy()
        
        data_temp['pp'] = (data_temp['high'] + data_temp['low'] + data_temp['close'])/3 
           
        data_temp['r1'] = 2 * data_temp['pp'] - data_temp['low']
        data_temp['s1'] = 2 * data_temp['pp'] - data_temp['high']
    
        data_temp['r2'] =  data_temp['pp'] + (data_temp['r1'] - data_temp['s1'])
        data_temp['s2'] =  data_temp['pp'] - (data_temp['r1'] - data_temp['s1'])
        
        data_temp[['pp', 'r1', 'r2', 's1', 's2']] = data_temp[['pp', 'r1', 'r2', 's1', 's2']].shift(1) 
        
        data_temp['pp_anterior'] = data_temp['pp'].shift(1)
        
        data_temp['ref_pp'] = -1
        data_temp['ref_pp'] = np.where(data_temp['pp'] > data_temp['pp'].shift(1), 1, data_temp['ref_pp'] )
        
        return data_temp
        
        
    