import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore")
import quantstats as qs
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC,abstractclassmethod
import statsmodels.api as sm
import itertools
from Marketstatus import LateralMarketIdentification
from ta.volatility import bollinger_hband,bollinger_lband,keltner_channel_hband,keltner_channel_lband
from ta.trend import ADXIndicator,macd_diff
from ta.momentum import rsi,stoch
from tqdm import tqdm
import time
class BaseStrategy(ABC):
    def __init__(self,
                 data,
                 base,
                 quote,
                 comision_margin,
                 date_init=datetime(2018,1,1),
                 date_final=datetime(2020,1,1),
                 candle_minutes=120,
                 optimize="sharpe",
                 maximum_volatility=1000,
                 **kwargs):
        
        self.time_serie_vec=data
        self.time_serie=data[0]
        self.quote=quote
        self.base=base
        self.optimize=optimize
        self.candle_minutes = candle_minutes
        self.periods=365*24*60/candle_minutes
        self.comision_margin={base:((1+comision_margin[base])**((365)/self.periods))-1,
                              quote:((1+comision_margin[quote])**((365)/self.periods))-1}
        self.date_init=date_init
        self.date_final=date_final
        self.maximum_volatility=maximum_volatility
        self.set_constants(**kwargs)
        self.name=self.get_strategy_name()
        self.pbounds=self.get_default_parameters_bounds()
        
        kwargs.setdefault("perc_to_inv",1)
        kwargs.setdefault("margin_long",0)
        kwargs.setdefault("margin_short",0)
        self.perc_to_inv = kwargs["perc_to_inv"]
        self.margin_short = kwargs["margin_short"]
        self.margin_long = kwargs["margin_long"]
        
    def set_constants(self,**kwargs):
        return
    
    @abstractclassmethod
    def get_strategy_name():
        pass
    
    @abstractclassmethod
    def get_default_parameters_bounds(self):
        pass
    
    def get_perc_to_inv(self):
        return self.perc_to_inv
    
    def get_margin_short(self):
        return self.margin_short
    
    def get_margin_long(self):
        return self.margin_long    
    
    def set_parameters_bounds(self,new_pbounds):
        return self.pbounds.update(new_pbounds)
        
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
        
    def function_to_optimize(self,**kwargs):
        #start_time=time.time()        
        result = np.array([self.get_output_per_sample(i) for i in itertools.product(np.arange(len(self.time_serie_vec)),[kwargs])])
        #print(time.time()-start_time)
        return result.mean()
    
    def get_output_per_sample(self,inputs):
        idf,kwargs=inputs
        output=0
        returns=self.get_returns(idf,**kwargs)
        volatility=qs.stats.volatility(returns,periods=self.periods)
        
        if (volatility==0) or (volatility>self.maximum_volatility) :
            if self.optimize=="sharpe":
                output+=(-3)
            elif self.optimize=="sortino":
                output+=(-6)
            elif self.optimize=="kelly": 
                output+=(-1)
            elif self.optimize=="return":
                output+=(0)
        elif self.optimize=="sharpe":
            output+=qs.stats.sharpe(np.log(1+returns),periods=self.periods)    
        elif self.optimize=="sortino":
            output+=qs.stats.sortino(np.log(1+returns),periods=self.periods)    
        elif self.optimize=="kelly": 
            if np.isnan(qs.stats.kelly_criterion(np.log(1+returns))) :
                output+=-1
            else:
                output+=qs.stats.kelly_criterion(np.log(1+returns))
        elif self.optimize=="return":
            output+=np.log(1+returns).mean()*self.periods
        
        return output
    
    def get_returns(self,idf,**kwargs):
        
        self.time_serie=self.time_serie_vec[idf]
        self.idf=idf
        #self.time_serie_long=self.time_serie_long_vec[idf]
        
        margin_long=self.get_margin_long()
        margin_short=self.get_margin_short()
        perc_to_inv=self.get_perc_to_inv()
        
        df=pd.DataFrame()
        df["pct"]=self.time_serie["close"].pct_change()
        df["action"]=self.get_actions(**kwargs)
        
        if (df["action"]==0).all():
            df["returns"]=0
            df=df.loc[self.date_init:self.date_final]
            return (df["returns"])
        
        df=df.loc[self.date_init:self.date_final]
        #apalancamiento y sus respectivas comisiones
        df["returns"] = np.where(df["action"] == 1,
                                 df["pct"]*(perc_to_inv+margin_long)-margin_long*self.comision_margin[self.quote],
                                 np.where(df["action"] == -1,
                                          (-1)*(df["pct"])*margin_short-margin_short*self.comision_margin[self.base],
                                          0))
        #comisiones de operatoria
        if self.quote == "USDT":  
            df.loc[((df["action"].shift(1)==0) & (df["action"]==1))|\
                   ((df["action"].shift(1)==1) & (df["action"]==0)),"returns"]-=\
                   0.001*(perc_to_inv+margin_long)
            df.loc[((df["action"].shift(1)==0) & (df["action"]==-1))|\
                   ((df["action"].shift(1)==-1) & (df["action"]==0)),"returns"]-=\
                   0.001*(margin_short)
            df.loc[((df["action"].shift(1)==1)& (df["action"]==-1))|\
                   ((df["action"].shift(1)==-1)& (df["action"]==1)),"returns"]-=\
                   0.001*(perc_to_inv+margin_long+margin_short)
        else:
            df.loc[((df["action"].shift(1)==0) & (df["action"]==1))|\
                   ((df["action"].shift(1)==1) & (df["action"]==0)),"returns"]-=\
                   0.002*(perc_to_inv+margin_long)
            df.loc[((df["action"].shift(1)==0) & (df["action"]==-1))|\
                   ((df["action"].shift(1)==-1) & (df["action"]==0)),"returns"]-=\
                   0.002*(margin_short)
            df.loc[((df["action"].shift(1)==1)& (df["action"]==-1))|\
                   ((df["action"].shift(1)==-1)& (df["action"]==1)),"returns"]-=\
                   0.002*(perc_to_inv+margin_long+margin_short)
        return df["returns"]
    
    
class HoldingStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "hold"
    
    def get_default_parameters_bounds(self):
        return {}
    
    def get_actions(self,**kwargs):
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        df["action"] = 1
        return df["action"]
    
class TwoMovingAverageStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "2sma"
    
    def get_default_parameters_bounds(self):
        return {"ma_short":(1,50),
                "ma_long":(10,300)}
    
    def get_actions(self,**kwargs):
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        ma_short = int(kwargs["ma_short"]) 
        ma_long = int(kwargs["ma_long"])
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        if (ma_short >= ma_long) : 
            df["action"] = 0
            return df["action"]
    
        df["ma_short"] = self.time_serie["close"].rolling(ma_short).mean()
        df["ma_long"] = self.time_serie["close"].rolling(ma_long).mean()
        df["action"] = 0
        df.loc[(df["ma_long"] > df["ma_short"]) ,"action"] = -1
        df.loc[(df["ma_short"] > df["ma_long"]),"action"] = 1 

        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()

        return df["action"]

class TwoMovingAverageLateralStrategy(BaseStrategy):
    
    def set_constants(self,**kwargs):
        self.lateral_identificator=[]
        print("setting lateral constants")
        for df in tqdm(self.time_serie_vec):
            df_l=df[(df.index.hour==0) & (df.index.minute==0)]
#            df_l=df_l.reindex(
#                    pd.date_range(freq=f"{self.candle_minutes}t",start=df.index[0],end=df.index[-1]))
#            df_l=df_l.ffill()
            self.lateral_identificator.append(LateralMarketIdentification(df,df_l,freq_short=f"{self.candle_minutes}t",freq_long="1d"))
        return
    
    def get_strategy_name(self):
        return "2sma_lat"
    
    def get_default_parameters_bounds(self):
        return {"ma_short":(1,50),
                "ma_long":(10,300),
                "mult_lateral":(.5,2)}
    
    def get_actions(self,**kwargs):
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        ma_short = int(kwargs["ma_short"]) 
        ma_long = int(kwargs["ma_long"])

        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        
        if (ma_short >= ma_long) : 
            df["action"] = 0
            return df["action"]
        
        df["ma_short"] = self.time_serie["close"].rolling(ma_short).mean()
        df["ma_long"] = self.time_serie["close"].rolling(ma_long).mean()

        df["action"] = 0

        df.loc[(df["ma_long"] > df["ma_short"]) ,"action"] = -1
        df.loc[(df["ma_short"] > df["ma_long"]),"action"] = 1
        
        df["lateral"] = self.lateral_identificator[self.idf].identify_lateral(kwargs["mult_lateral"])
        df.loc[df["lateral"]==1,"action"]=0
        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()
        

        return df["action"]

class ThreeMovingAverageStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "3sma"
    
    def get_default_parameters_bounds(self):
        return {"ma_short_buy":(1,50),
                "ma_short_sell":(1,50),
                "ma_long":(30,300)}
    
    def get_actions(self,**kwargs):
 
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """

        ma_short = int(kwargs["ma_short_buy"]) 
        ma_medium = int(kwargs["ma_short_sell"]) 
        ma_long = int(kwargs["ma_long"])
       
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        
        if (ma_short >= ma_long) or (ma_medium >= ma_long) : 
            df["action"] = 0
            return df["action"]

        df["ma_short"] = self.time_serie["close"].rolling(ma_short).mean()
        df["ma_medium"] = self.time_serie["close"].rolling(ma_medium).mean()
        df["ma_long"] = self.time_serie["close"].rolling(ma_long).mean()

        df["action"] = 0

        df.loc[(df["ma_short"] > df["ma_long"]),"action"] = 1 
        df.loc[ (df["ma_medium"] < df["ma_long"]) ,"action"] = -1
        
        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()
        
        return df["action"]
    
class ThreeMovingAverageLateralStrategy(TwoMovingAverageLateralStrategy):
    
    def get_strategy_name(self):
        return "3sma_lat"
    
    def get_default_parameters_bounds(self):
        return {"ma_short_buy":(1,50),
                "ma_short_sell":(1,50),
                "ma_long":(30,300),
                "mult_lateral":(.5,2)}
    
    def get_actions(self,**kwargs):
 
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        ma_short = int(kwargs["ma_short_buy"]) 
        ma_medium = int(kwargs["ma_short_sell"]) 
        ma_long = int(kwargs["ma_long"])
              
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        
        if (ma_short >= ma_long) or (ma_medium >= ma_long) : 
            df["action"] = 0
            return df["action"]


        df["ma_short"] = self.time_serie["close"].rolling(ma_short).mean()
        df["ma_medium"] = self.time_serie["close"].rolling(ma_medium).mean()
        df["ma_long"] = self.time_serie["close"].rolling(ma_long).mean()

        df["action"] = 0

        df.loc[(df["ma_short"] > df["ma_long"]),"action"] = 1 
        df.loc[(df["ma_medium"] < df["ma_long"]) ,"action"] = -1

        df["lateral"]=self.lateral_identificator[self.idf].identify_lateral(kwargs["mult_lateral"]).shift(1)
        df.loc[df["lateral"]==1,"action"]=0
        
        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()
        
        return df["action"]

class ThreeMovingAverageAltStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "3sma_alt"
    
    def get_default_parameters_bounds(self):
        return {"ma_short":(1,50),
                "ma_medium":(10,200),
                "ma_long":(30,400)}
    
    def get_actions(self,**kwargs):
        
        ma_short=int(kwargs["ma_short"])
        ma_medium=int(kwargs["ma_medium"])
        ma_long=int(kwargs["ma_long"])
        
        df=pd.DataFrame()
        df["close"] = self.time_serie["close"]
        
        if (ma_short >= ma_long) or (ma_short >= ma_medium) or (ma_medium >= ma_long) : 
            df["action"] = 0
            return df["action"]

        df["sma_short"]=self.time_serie["close"].rolling(ma_short).mean()
        df["sma_medium"]=self.time_serie["close"].rolling(ma_medium).mean()
        df["sma_long"]=self.time_serie["close"].rolling(ma_long).mean()
    
        df["action"]=0
        
        df.loc[(df["sma_short"]>df["sma_long"])  & (df["sma_short"]>df["sma_medium"]) ,"action"]=1
        df.loc[(df["sma_short"]<df["sma_long"])  & (df["sma_short"]<df["sma_medium"]),"action"]=-1
        
        
        
        return df["action"].shift(1).replace(0,np.nan).ffill()

class ThreeMovingAverageAltLateralStrategy(TwoMovingAverageLateralStrategy):
    
    def get_strategy_name(self):
        return "3sma_lat_atl"
    
    def get_default_parameters_bounds(self):
        return {"ma_short":(1,50),
                "ma_medium":(10,200),
                "ma_long":(30,400),
                "mult_lateral":(.5,2)}
    
    def get_actions(self,**kwargs):
        
        ma_short=int(kwargs["ma_short"])
        ma_medium=int(kwargs["ma_medium"])
        ma_long=int(kwargs["ma_long"])
        
        df=pd.DataFrame()
        df["close"] = self.time_serie["close"]
    
        if (ma_short >= ma_long) or (ma_short >= ma_medium) or (ma_medium >= ma_long): 
            df["action"] = 0
            return df["action"]

        df["sma_short"]=self.time_serie["close"].rolling(ma_short).mean()
        df["sma_medium"]=self.time_serie["close"].rolling(ma_medium).mean()
        df["sma_long"]=self.time_serie["close"].rolling(ma_long).mean()
    
        df["action"]=0
        
        df.loc[(df["sma_short"]>df["sma_long"])  & (df["sma_short"]>df["sma_medium"]) ,"action"]=1
        df.loc[(df["sma_short"]<df["sma_long"])  & (df["sma_short"]<df["sma_medium"]),"action"]=-1
        df["lateral"]=self.lateral_identificator[self.idf].identify_lateral(kwargs["mult_lateral"]).shift(1)
        
        df.loc[df["lateral"]==1,"action"]=0
        
        return df["action"].shift(1).replace(0,np.nan).ffill()
        
        
class VWAPvsSMAStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "vwap_sma"
    
    def get_default_parameters_bounds(self):
        return {"lecturas":(5,2000),
                "porcentaje":(0,0.5)}
    
    def get_actions(self,**kwargs):
        lecturas=int(kwargs["lecturas"])
        porcentaje=kwargs["porcentaje"]
        df=pd.DataFrame()
        df["vwap"]=((self.time_serie["volume"]*self.time_serie["close"]).rolling(lecturas).mean()/(self.time_serie["volume"]).rolling(lecturas).mean())
        df["sma"]=(self.time_serie["close"]).rolling(lecturas).mean()
        
        df["porcentaje"]=abs(np.log(df["vwap"]/df["sma"]))*100
        
        df["action"]=0
        df["action"]=np.where((df["porcentaje"]>porcentaje)&(df["sma"]<df["vwap"])
                        ,1,
                        np.where((df["porcentaje"]>porcentaje)&(df["sma"]>df["vwap"]),
                                 -1
                                 ,0
                                 )
                        )

        
        df["action"]=df["action"].shift(1).replace(0,np.nan).ffill()

        return df["action"]


class VWAPvsSMALateralStrategy(TwoMovingAverageLateralStrategy):
    
    def get_strategy_name(self):
        return "vwap_sma_lateral"
    
    def get_default_parameters_bounds(self):
        return {"lecturas":(5,2000),
                "porcentaje":(0,0.5),
                "mult_lateral":(.5,2)}
    
    def get_actions(self,**kwargs):
        lecturas=int(kwargs["lecturas"])
        porcentaje=kwargs["porcentaje"]
        df=pd.DataFrame()
        df["vwap"]=((self.time_serie["volume"]*self.time_serie["close"]).rolling(lecturas).mean()/(self.time_serie["volume"]).rolling(lecturas).mean())
        df["sma"]=(self.time_serie["close"]).rolling(lecturas).mean()
        
        df["porcentaje"]=abs(np.log(df["vwap"]/df["sma"]))*100
        
        df["action"]=0
        df["action"]=np.where((df["porcentaje"]>porcentaje)&(df["sma"]<df["vwap"])
                        ,1,
                        np.where((df["porcentaje"]>porcentaje)&(df["sma"]>df["vwap"]),
                                 -1
                                 ,0
                                 )
                        )
        df["lateral"]=self.lateral_identificator[self.idf].identify_lateral(kwargs["mult_lateral"]).shift(1)
        df.loc[df["lateral"]==1,"action"]=0
        df["action"]=df["action"].shift(1).replace(0,np.nan).ffill()

        return df["action"]

class PivotPointsStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "pivot_points"
    
    def get_default_parameters_bounds(self):
        return {}
    
    def get_actions(self,**kwargs):
        
        df=self.get_pivots(self.time_serie_long)
        df=df.asfreq(self.time_serie.index.freq).ffill()
        lista = ['pp', 'r1', 's1', 'r2', 's2', 'ref_pp', 'pp_anterior']
        
        for col in lista:
            self.time_serie[col]=df[col]
        
        
        df=self.time_serie.copy()
        
        df['trend'] =  np.where(df['close']>(df['s2']+df['s1'])/2,1,0)
        indice = df[ (df['close'].shift(1) > df['r1']) & (df['close'] < df['r1'])].index
        df['trend'].loc[indice]=0

        df["action"]=0
        df.loc[df['trend'] ==1 ,"action"]=1
        df.loc[df['trend'] ==0,"action"]=-1 
        
        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()

        return df["action"]  
    
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
        data_temp['ref_pp'] = np.where(data_temp['pp'] > data_temp['pp_anterior'], 1, data_temp['ref_pp'] )
        
        return data_temp

class SlopeStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "slope"
    
    def get_default_parameters_bounds(self):
        return {"short":(1,40),
                "medium":(10,200),
                "long":(30,400),
                "window":(5,21),
                "biases":(3,6)}
    
    def get_actions(self,**kwargs):
        ma_short = int(kwargs["short"]) 
        ma_medium = int(kwargs["medium"])
        ma_long = int(kwargs["long"])
        window=int(kwargs["window"])
        biases=int(kwargs["biases"])
        
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        
        #Filtro para no tener medias cortas mas largas que las largas
        if (ma_short >= ma_long) or (ma_short >= ma_medium) or (ma_medium >= ma_long) : 
            df["action"] = 0
            return df["action"]
        
        #Calculo de medias exponenciales
        df["ma_short"] = df["close"].ewm(span=ma_short, adjust=False).mean()
        df["ma_medium"] = df["close"].ewm(span=ma_medium, adjust=False).mean()
        df["ma_long"] = df["close"].ewm(span=ma_long, adjust=False).mean()       
        df['ma_dif'] = abs( df['ma_medium'] - df['ma_long'] )
        
        #Calculos de pendiente con Numpy
        signal = self.get_signals(df['ma_dif'].values, window)
        signal_array = np.zeros(window)
        signal_array = np.append(signal_array, signal)
        
        #Columna de senales puras, estructuro la data para analizarla.
        df['signal'] = signal_array
        #Columna de senales acumuladas
        df['signal_acum'] = df['signal'].groupby((df['signal'] != df['signal'].shift()).cumsum()).cumcount()+1

        # Veamos cuando la media corta es cruzada de abajo hacia arriba por el precio 
        df['signal_short'] = np.where(df.close > df.ma_short, 1 , 0 )
        df['signal_short'] = np.where(df.close < df.ma_short, -1 , df.signal_short)

        #Buscando los retornos de la estrategia    
        df["action"] = 0
        df.loc[ (df.signal_acum >= biases) & (df.signal_short == -1), 'action'] = -1    
        df.loc[ (df.signal_acum >= biases) & (df.signal_short == 1), 'action' ] = 1     

        #Relleno falsas salidas. Los valores en 0, los reemplazo por el anterior
        df['action'] = df['action'].replace(to_replace=0, method='ffill')
        df["action"] = df["action"].shift(1).replace(0,np.nan).ffill()
        
        return df["action"]


    def get_signals(self,df_col,window): 
        data = np.empty(0) #Inicio numpy array vacio
        for i in range(len(df_col)- window):
            y = df_col[i : window+i]#formato Numpy
            x = np.arange(1, window+1)
            
            #FITTING
            # y = a*x+b
            x = sm.add_constant(x)
            model = sm.OLS(y,x).fit()
            b, a = model.params
            
            #PREDICTION
            x_ = window+1
            y_ = df_col[window+i]
            y_pred = round( a*x_ + b, 4)
            
            #OUTPUT
            if y_ >= y_pred :
                signal = 1 
            else:
                signal = -1
            data = np.append(data, signal) #Armo numpy array de seniales
                
        return data
    
    
class ChandelierExitStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "chandelier_exit"
    
    def get_default_parameters_bounds(self):
        return {"mean":(10,50),
                "chand_window":(10,70),
                "mult_high":(1,4),
                "mult_low":(1,4)}
    
    def calculate_true_range(self, df):
        '''
        El máximo de entre:
            * Diferencia entre el high y el low del mismo dia,
            * Diferencia entre el high de hoy menos el cierre de ayer, y
            * Diferencia entre el low de hoy menos el cierre de ayer.
        '''
        df['tr1'] = df["high"] - df["low"]
        df['tr2'] = abs(df["high"] - df["close"].shift(1))
        df['tr3'] = abs(df["low"] - df["close"].shift(1))
        df['TR'] = df[['tr1','tr2','tr3']].max(axis=1)
        df.loc[df.index[0],'TR'] = 0
        return df

    def calculate_average_true_range(self, df, mean):
        '''
        Para el primer ATR, calcula el promedio de los ultimos mean periodos.
        Para los siguientes ATRs, multiplica el ultimo ATR calculado por mean-1,
        le suma el ultimo TR calculado y luego divide todo por mean.
        
        '''
        df = self.calculate_true_range(df)
        df['ATR'] = 0
        df.loc[df.index[mean],'ATR'] = round( df.loc[df.index[1:mean+1],"TR"].rolling(window=mean).mean()[-1], 4)
        const_atr = (mean-1)/mean
        const_tr = 1/mean
        ATR=df["ATR"].values
        TR=df["TR"].values
        for index in range(mean+1, len(df)):
            ATR[index]=ATR[index-1]*const_atr+TR[index]*const_tr
            #df.loc[df.index[index],'ATR'] = (df.loc[df.index[index-1],'ATR'] * const_atr+ df.loc[df.index[index],'TR'] * const_tr) 
        df["ATR"]=ATR
        return df
    
    def calculate_chandelier_exits(self, df, mean, chand_window, mult_high, mult_low):
        mean=int(mean)
        chand_window=int(chand_window)        
        df = self.calculate_average_true_range(df, mean)
        df["chandelier_high"] = df['close']
        df["chandelier_low"] = df['close']
        df.loc[df.index[chand_window+1:],"chandelier_low"] = df.loc[df.index[chand_window+1:],"low"].rolling(window=chand_window).min() + mult_low * df["ATR"][chand_window+1:]
        df.loc[df.index[chand_window+1:],"chandelier_high"] = df.loc[df.index[chand_window+1:],"high"].rolling(window=chand_window).max() - mult_high * df["ATR"][chand_window+1:]
        return df
    
    def get_actions(self, **kwargs):
        mean = int(kwargs["mean"]) 
        chand_window = int(kwargs["chand_window"])
        mult_high = int(kwargs["mult_high"])
        mult_low=int(kwargs["mult_low"])
        df = self.time_serie.copy()
        self.calculate_chandelier_exits(df, mean, chand_window, mult_high, mult_low)
        
        #defino la acciones a realizar
        df["action"] = 0
            #COMPRA: El precio tiene que estar por encima de las dos lineas
        df["action"] = np.where(df["close"] > df["chandelier_low"],
                                  1,
                                  df["action"])
            #VENTA: El precio tiene que estar por debajo de chand_low
        df["action"] = np.where(df["close"] < df["chandelier_high"],
                                  -1,
                                  df["action"])
        df["action"] = df["action"].shift(1) #ACA SI CONSIDERO SHIFTS DE LA BASE PARA OPERAR EN OPEN SIGUIENTE
        #saco la primer obs, ahora vacia
    
        return df["action"]
    
    
class AccumulatedVolumeStrategy(BaseStrategy):
    
    def get_strategy_name(self):
        return "accumulated_volume"
    
    def get_default_parameters_bounds(self):
        return {'obs_accum': (1, 9), 
                'target_window': (1, 10),
                'volume_mean': (1, 200),
                'volume_mean_shift': (1, 3),
                'bb_periods': (10, 30) ,
                'std_bb': (2, 3.5)}
    
    def set_market_direction(self, df, bb_periods, std_bb):
        bb_high = bollinger_hband(df['close'], n=bb_periods, ndev=std_bb, fillna=False)
        bb_low = bollinger_lband(df['close'], n=bb_periods, ndev=std_bb, fillna=False)
        df['direction'] = 0
        df['direction'] = np.where(df['close'] > bb_high, 1, df['direction'])
        df['direction'] = np.where(df['close'] < bb_low, -1, df['direction'])           
        return df

    def get_actions(self, **kwargs):
        obs_accum = int(kwargs['obs_accum'])
        target_window = int(kwargs['target_window'])
        volume_mean = int(kwargs['volume_mean'])
        volume_mean_shift = int(kwargs['volume_mean_shift'])
        bb_periods = int(kwargs['bb_periods'])
        std_bb = int(kwargs['std_bb']) 
        
        df = self.time_serie.copy()   
        df['volume_acum'] = df['volume'].rolling(obs_accum).sum().fillna(0)
        df['volume_mean'] = df['volume_acum'].ewm(span=volume_mean, adjust=False).mean().fillna(0) * volume_mean_shift       
        df = self.set_market_direction(df, bb_periods, std_bb)
        df['action'] = 0    
        df['action'] = np.where((df['volume_acum'] > df['volume_mean']), df['direction'], 0)
        df = df.reset_index()
        signals = df.loc[(df['action'] != df['action'].shift(1)) & (df['action'] != 0)]        

        for index in signals.index:
            df.loc[index:index+target_window,'action'] = signals.loc[index,'action']
        df = df.set_index("index")
        return df["action"].shift(1)
         
    
class TwoStrategiesCombination(BaseStrategy):
    
    def __init__(self,
                 data,
                 base,
                 quote,
                 comision_margin,
                 base_strategy=TwoMovingAverageStrategy,
                 exp_strategy=ChandelierExitStrategy,
                 date_init=datetime(2018,1,1),
                 date_final=datetime(2020,1,1),
                 candle_minutes=120,
                 optimize="sharpe",
                 maximum_volatility=1000,
                 **kwargs):
        """
        Class to combine two strategies
        """
        
        self.time_serie_vec=data
        self.time_serie=data[0]
        self.quote=quote
        self.base=base
        self.optimize=optimize
        self.base_strategy = base_strategy(data,
                 base, quote,comision_margin,
                 date_init=date_init, date_final=date_final,
                 candle_minutes=candle_minutes,
                 optimize=optimize,
                 maximum_volatility=maximum_volatility,
                 **kwargs)
        self.exp_strategy = exp_strategy(data,
                 base, quote,comision_margin,
                 date_init=date_init, date_final=date_final,
                 candle_minutes=candle_minutes,
                 optimize=optimize,
                 maximum_volatility=maximum_volatility,
                 **kwargs)
        
        self.periods=365*12*60/candle_minutes
        self.comision_margin={base:((1+comision_margin[base])**(self.periods/(365*24)))-1,
                              quote:((1+comision_margin[quote])**(self.periods/(365*24)))-1}
        self.date_init=date_init
        self.date_final=date_final
        self.maximum_volatility=maximum_volatility
        self.set_constants(**kwargs)
        self.name=self.get_strategy_name()
        self.pbounds=self.get_default_parameters_bounds()
        kwargs.setdefault("perc_to_inv",1)
        kwargs.setdefault("margin_long",0)
        kwargs.setdefault("margin_short",0)
        self.perc_to_inv = kwargs["perc_to_inv"]
        self.margin_short = kwargs["margin_short"]
        self.margin_long = kwargs["margin_long"]
        
    def get_strategy_name(self):
        name = f'{self.base_strategy.name}_{self.exp_strategy.name}'
        return name
        
    def get_default_parameters_bounds(self):
        
        parameters_bounds={}
        parameters_bounds_base = self.base_strategy.get_default_parameters_bounds()
        parameters_bounds_exp = self.exp_strategy.get_default_parameters_bounds()
        intersection = set(parameters_bounds_base).intersection(parameters_bounds_exp)
        
        if intersection:
            print(f"Warning: {intersection} keys are present in the two strategy parameter bounds")
        parameters_bounds.update(parameters_bounds_base)
        parameters_bounds.update(parameters_bounds_exp)
        
        return parameters_bounds
    
    def get_actions(self, **kwargs):
        df = self.time_serie.copy()
        self.base_strategy.time_serie=df
        #self.base_strategy.idf=self.idf
        self.exp_strategy.time_serie=df
        #self.exp_strategy.idf=self.idf
        
        df['action_base'] = self.base_strategy.get_actions(**kwargs)
        df['action_exp'] = self.exp_strategy.get_actions(**kwargs)
        
        #Acciones combinadas
        df['action'] = 0
        df.loc[ (df["action_base"] == 1) & (df["action_exp"] == 1), "action"] = 1
        df.loc[ (df["action_base"] == 1) & (df["action_exp"] == 0), "action"] = 1
        df.loc[ (df["action_base"] == 1) & (df["action_exp"] == -1), "action"] = 0
        df.loc[ (df["action_base"] == 0) & (df["action_exp"] == 1), "action"] = 1
        df.loc[ (df["action_base"] == 0) & (df["action_exp"] == 0), "action"] = 0
        df.loc[ (df["action_base"] == 0) & (df["action_exp"] == -1), "action"] = -1
        df.loc[ (df["action_base"] == -1) & (df["action_exp"] == 1), "action"] = 0
        df.loc[ (df["action_base"] == -1) & (df["action_exp"] == 0), "action"] = -1
        df.loc[ (df["action_base"] == -1) & (df["action_exp"] == -1), "action"] = -1
        
        return df["action"]
    
class MACDRSIStochStrategy(BaseStrategy):
    def get_default_parameters_bounds(self):
        return {'macd_slow' : (20,30),
                'macd_fast' : (8,15),
                'macd_signal' : (7,13),
                'rsi_n' : (11,17),
                'stoch_n' : (11,17),
                'stoch_dn' : (1,6),
                'stoch_requp' : (60,90),
                'stoch_reqdown' : (10,40),
                'rsi_requp' : (60,90),
                'rsi_reqdown' : (10,40),
                'take_profit' : (5,30),
                'stop_loss' : (5,15)}
    
    def get_strategy_name(self):
        return "macdrsistoch"
    
    def get_actions(self,**kwargs):
        
        macd_slow = int(kwargs["macd_slow"])
        macd_fast = int(kwargs["macd_fast"])
        macd_signal = int(kwargs["macd_signal"])
        rsi_n = int(kwargs["rsi_n"])
        stoch_n = int(kwargs["stoch_n"])
        stoch_dn = int(kwargs["stoch_dn"])
        stoch_requp = kwargs["stoch_requp"]
        stoch_reqdown = kwargs["stoch_reqdown"]
        rsi_requp = kwargs["rsi_requp"]
        rsi_reqdown = kwargs["rsi_reqdown"]
        take_profit = 1+(kwargs["take_profit"]/100.)
        stop_loss = 1-(kwargs["stop_loss"]/100.)
                
        df=self.time_serie.copy()
        pct = 1+df["close"].pct_change().values
        MACD = macd_diff(df["close"],n_slow=macd_slow,n_fast=macd_fast,n_sign=macd_signal,fillna=False).values
        RSI = rsi(df["close"],n=rsi_n,fillna=False).values
        Stoch = stoch(df["high"],df["low"],df["close"],n=stoch_n,d_n=stoch_dn,fillna=False).values
        
        action = np.zeros(len(df))
        action[(RSI<rsi_reqdown) & (Stoch<stoch_reqdown) & (MACD<0)] = 1
        action[(RSI>rsi_requp) & (Stoch>stoch_requp) & (MACD>0)] = -1

        counting = False
        short=False
        cumprod = 1
        for i in range(len(df)):
            
            if action[i] == 1:
                counting = True
                short=False
                cumprod = 1
            elif action[i] == -1:
                counting = True
                short=True
                cumprod = 1
            
            if cumprod>take_profit:
                counting = False
                cumprod = 1
                action[i]=0
            elif cumprod<stop_loss:
                counting = False
                cumprod = 1
                action[i]=0
                
            if counting :
                if short:
                    cumprod *= (-1)*pct[i]
                    action[i] = -1
                else:
                    cumprod*=pct[i]
                    action[i] = 1
        del pct,MACD,RSI,Stoch,cumprod
        
        return pd.Series(action,index=df.index).shift(1)
                
    
####################################################### María    

class ADXStrategy(BaseStrategy):
    
    def set_constants(self,**kwargs):
        
        self.online = kwargs.setdefault("onlineADX",True)
        self.length_adx_bound = kwargs.setdefault("length_adx_bound",30)
        if self.online == True:
            return
        
        number_of_samples = len(self.time_serie_vec)
        self.adxI = np.empty((number_of_samples,self.length_adx_bound),dtype=object)
        for sample in tqdm(range(number_of_samples)):
            df=self.time_serie_vec[sample]
            for length_adx in range(1,self.length_adx_bound+1):
                self.adxI[sample][length_adx-1] = ADXIndicator(df['high'],df['low'],df["close"],length_adx,False)
        del df
        return
    
    def get_strategy_name(self):
        return "adx"
    
    def get_default_parameters_bounds(self):
        return {'bb_periods':(6,24), 
                'kc_periods':(6,24),
                'std_bb':(1,3),
                'std_kc':(1,3),
                'rsi_periods':(6,24),
                'adx_periods':(1,self.length_adx_bound),
                'stoch_rsi_upper':(25,45),
                'stoch_rsi_low':(25,45),
                'adx_neg_bound':(25,45),
                'adx_bound':(25,45)}
    
    def calculate_bollinger_band(self, df,col, periodo, desviacion): 
        df_temp = df.copy()
        df_temp['bb_medium'] = df_temp[col].rolling(window=periodo).mean()
        df_temp['bb_high'] =  df_temp['bb_medium'] + desviacion * df_temp[col].rolling(window=periodo).std()
        df_temp['bb_low'] =  df_temp['bb_medium'] - desviacion * df_temp[col].rolling(window=periodo).std()
        return df_temp
    
    def calculate_keltner_channel(self,df,col, periodo, desviacion): 
        df_temp = df.copy()
        df_atr = self.calculate_atr(df_temp,periodo)    
        df_temp['kc_medium'] = df_temp[col].ewm(span=periodo, adjust=False).mean()
        df_temp['kc_high'] =  df_temp['kc_medium'] + desviacion * df_atr.atr
        df_temp['kc_low'] =  df_temp['kc_medium'] - desviacion * df_atr.atr
        return df_temp

    def calculate_atr(self,df,periodo):
        df_temp = df.copy()
        df_temp['h-l'] = df_temp['high'] - df['low']
        df_temp['h-c'] = df_temp['high'] - df['close'].shift(1)
        df_temp['l-c'] = df_temp['low'] - df['close'].shift(1)
        df_temp['max'] = df_temp[['h-l', 'h-c', 'l-c']].max(axis=1)
        df_temp['atr'] = df_temp['max'].rolling(window=periodo).mean()
        return df_temp

    def calculate_rsi(self,df, col, periodo):
        df_temp = df[[col]].copy()
        df_temp['dif'] = df_temp[col] - df_temp[col].shift(1)
        df_temp['up'] = np.where(df_temp['dif'] > 0 , df_temp['dif'] , 0 )
        df_temp['down'] = np.where(df_temp['dif'] < 0 , df_temp['dif'] , 0 )
        df_temp['up_mean'] = df_temp[['up']].rolling(window=periodo).mean()
        df_temp['down_mean'] = df_temp[['down']].rolling(window=periodo).mean()
        df_temp['rs'] = df_temp['up_mean'] / abs(df_temp['down_mean'])
        df_temp['rsi'] = 100 - 100/(1+df_temp['rs'])    
        return df_temp[['rsi']]

    def calculate_stoch_rsi(self,df, col, periodo, n_wind=3):
        df_temp = self.calculate_rsi(df.copy(), col, periodo)
        df_temp = df_temp.dropna()
        df_temp['rsi_min'] = df_temp['rsi'].rolling(window=periodo).min()
        df_temp['rsi_max'] = df_temp['rsi'].rolling(window=periodo).max()
        df_temp['stoch_rsi'] = 100 * (df_temp['rsi'] - df_temp['rsi_min']) / (df_temp['rsi_max'] - df_temp['rsi_min'])    
        df_temp['stoch_rsi_sma'] = df_temp['stoch_rsi'].rolling(window=n_wind).mean()
        return df_temp[['stoch_rsi', 'stoch_rsi_sma']]
       
    def get_df_strategy(self, length_adx, length_bb, std_bb, length_kc, std_kc, length_rsi):

        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        df["low"] = self.time_serie["low"]
        df["high"] = self.time_serie["high"]

        # 
        if df.loc[df.index[0],'close'] < 1: 
            df = 1000*df

        #Calculate pct_change
        df["close_pct_change"]=self.time_serie["close"].pct_change()

        #Calculate ADX Indicator
        if self.online == True:
            adxI = ADXIndicator(df['high'],df['low'],df["close"],length_adx,False)
        else:
            adxI = self.adxI[self.idf][length_adx-1]
        df['di_up'] = adxI.adx_pos()
        df['di_down'] = adxI.adx_neg()
        df['adx'] = adxI.adx()

        #Calculate BB
        data_bollinger_bands = self.calculate_bollinger_band(df, "close", length_bb, std_bb) 

        #Calculate KC
        data_keltner_bands = self.calculate_keltner_channel(df, "close", length_kc, std_kc) 

        #Calculate Stochastic RSI
        data_stoch_rsi = self.calculate_stoch_rsi(df, "close", length_rsi, 2) #12

        #Concatenate data
        data_str_bb = data_bollinger_bands[['close', 'close_pct_change', 'bb_medium', 'bb_high', 'bb_low']] 
        data_str_kc = data_keltner_bands[['kc_medium', 'kc_high', 'kc_low']] 
        data_str_rsi = data_stoch_rsi[['stoch_rsi', 'stoch_rsi_sma']] 

        data_strategy = pd.concat([pd.concat([data_str_bb, data_str_kc], axis=1), data_str_rsi], axis=1)
        data_strategy = pd.concat([data_strategy, df[['adx', 'di_up', 'di_down']]], axis=1)

        return data_strategy

    def get_actions(self,**kwargs):
        """
        Tiene como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        length_bb=int(kwargs['bb_periods'])
        length_kc=int(kwargs['kc_periods'])
        std_bb=kwargs['std_bb']
        std_kc=kwargs['std_kc']
        length_rsi=int(kwargs['rsi_periods'])
        length_adx=int(kwargs['adx_periods'])
        stoch_rsi_upper=kwargs["stoch_rsi_upper"]
        stoch_rsi_low=kwargs["stoch_rsi_low"]
        di_down_bound = kwargs['adx_neg_bound']
        adx_bound = kwargs['adx_bound']

        df_temp = self.get_df_strategy(length_adx, length_bb, std_bb, length_kc, std_kc, length_rsi)

        #Buy
        df_temp['action'] = np.where((df_temp['stoch_rsi']>stoch_rsi_upper)\
                                & (df_temp['adx']>df_temp['adx'].shift(1))\
                                & (df_temp['di_down']<df_temp['di_up'])\
                                ,1, 0) 

        #Sell
        df_temp['action'] = np.where( ((df_temp['close']<df_temp['bb_low'])|(df_temp['close']<df_temp['kc_low']))\
                                &(df_temp['stoch_rsi']<stoch_rsi_low)\
                                &(df_temp['di_down']>di_down_bound)\
                                &(df_temp['adx']>adx_bound)\
                                & (df_temp['di_down']>df_temp['di_up']),\
                                 -1, df_temp['action'] )  
        
        if (df_temp["action"]==0).all():
            return df_temp["action"]
        else:
            df_temp['action'] = df_temp['action'].replace(0,np.nan)
            df_temp['action'] = df_temp['action'].fillna(method='ffill').shift(1)

        return df_temp["action"]


class ADXPPStrategy(ADXStrategy):
    
    def get_strategy_name(self):
        return "adx_pp"
    
    def get_default_parameters_bounds(self):
        return {'bb_periods':(6,24), 
                'kc_periods':(6,24),
                'std_bb':(1,3),
                'std_kc':(1,3),
                'rsi_periods':(6,24),
                'adx_periods':(1,self.length_adx_bound),
                'stoch_rsi_upper':(25,45),
                'stoch_rsi_low':(25,45),
                'adx_neg_bound':(25,45),
                'adx_bound':(25,45)}

    def get_pivots(self,df):
        data_temp = df.copy()
        data_temp['pp'] = (data_temp['high'] + data_temp['low'] + data_temp['close'])/3 
        data_temp['r1'] = 2 * data_temp['pp'] - data_temp['low']
        data_temp['s1'] = 2 * data_temp['pp'] - data_temp['high']
        data_temp['r2'] =  data_temp['pp'] + (data_temp['r1'] - data_temp['s1'])
        data_temp['s2'] =  data_temp['pp'] - (data_temp['r1'] - data_temp['s1'])
        data_temp[['pp', 'r1', 'r2', 's1', 's2']] = data_temp[['pp', 'r1', 'r2', 's1', 's2']].shift(1) 
        return data_temp

    def complete_df_with_pivot_points(self,df_period_long,df_period_short, cols):
        df_temp = pd.DataFrame(data=0,index=df_period_short.index, columns = cols)
        df_temp[cols[0]].loc[df_period_long.index] = df_period_long[cols[0]]
        df_temp[cols[1]].loc[df_period_long.index] = df_period_long[cols[1]]
        df_temp = df_temp.replace(0,np.nan)
        df_temp = df_temp.fillna(method='ffill')
        df_temp_ = pd.concat([df_period_short,df_temp], axis=1)
        del df_temp
        return df_temp_

    def get_data_one_day(self,df):
        index_1d = pd.date_range(start=df.index[0], end=df.index[-1], freq='1d')
        df_1d = pd.DataFrame(columns= ['open', 'close', 'high', 'low'], index=index_1d)
        df_1d['high'] = df['high'].resample('1D').max()
        df_1d['low'] = df['low'].resample('1D').min()
        df_1d['open'] = df.loc[df_1d.index]['open']
        df_1d['close'] = df.shift(1)['close'].loc[index_1d].shift(-1)
        df_1d = df_1d.fillna(method='ffill') 
        return df_1d
    
    def get_df_strategy(self, length_adx, length_bb, std_bb, length_kc, std_kc, length_rsi):
        df = pd.DataFrame()
        df["close"] = self.time_serie["close"]
        df["open"] = self.time_serie["open"]
        df["low"] = self.time_serie["low"]
        df["high"] = self.time_serie["high"]

        # 
        if df.loc[df.index[0],'close'] < 1: 
            df = 1000*df

        #Calculate pct_change
        df["close_pct_change"]=self.time_serie["close"].pct_change()

        #Calculate ADX Indicator
        if self.online == True:
            adxI = ADXIndicator(df['high'],df['low'],df["close"],length_adx,False)
        else:
            adxI = self.adxI[self.idf][length_adx-1]
        df['di_up'] = adxI.adx_pos()
        df['di_down'] = adxI.adx_neg()
        df['adx'] = adxI.adx()

        #Calculate BB
        data_bollinger_bands = self.calculate_bollinger_band(df, "close", length_bb, std_bb) 

        #Calculate KC
        data_keltner_bands = self.calculate_keltner_channel(df, "close", length_kc, std_kc) 

        #Calculate Stochastic RSI
        data_stoch_rsi = self.calculate_stoch_rsi(df, "close", length_rsi, 2) #12

        #Concatenate data
        data_str_bb = data_bollinger_bands[['close', 'close_pct_change', 'bb_medium', 'bb_high', 'bb_low']] 
        data_str_kc = data_keltner_bands[['kc_medium', 'kc_high', 'kc_low']] 
        data_str_rsi = data_stoch_rsi[['stoch_rsi', 'stoch_rsi_sma']] 

        data_strategy = pd.concat([pd.concat([data_str_bb, data_str_kc], axis=1), data_str_rsi], axis=1)
        data_strategy = pd.concat([data_strategy, df[['adx', 'di_up', 'di_down', 'low']]], axis=1)

        #Calculate Pivot Points
        df_1d = self.get_data_one_day(df)
        df_temp_1d = self.get_pivots(df_1d) 
        cols = ['pp', 's1']
        data_strategy = self.complete_df_with_pivot_points(df_temp_1d, data_strategy, cols) 
        data_strategy[cols] = data_strategy[cols].replace(0, np.nan)
        data_strategy[cols] = data_strategy[cols].fillna(method = 'ffill')        
        return data_strategy

    def get_actions(self,**kwargs):
        """
        Tiene como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        length_bb=int(kwargs['bb_periods'])
        length_kc=int(kwargs['kc_periods'])
        std_bb=kwargs['std_bb']
        std_kc=kwargs['std_kc']
        length_rsi=int(kwargs['rsi_periods'])
        length_adx=int(kwargs['adx_periods'])
        stoch_rsi_upper=kwargs["stoch_rsi_upper"]
        stoch_rsi_low=kwargs["stoch_rsi_low"]
        di_down_bound = kwargs['adx_neg_bound']
        adx_bound = kwargs['adx_bound']

        df_temp = self.get_df_strategy(length_adx, length_bb, std_bb, length_kc, std_kc, length_rsi)

        #Buy
        df_temp['action'] = np.where((df_temp['stoch_rsi']>stoch_rsi_upper)\
                                & (df_temp['adx']>df_temp['adx'].shift(1))\
                                & (df_temp['di_down']<df_temp['di_up'])\
                                ,1, 0)

        #Sell
        df_temp['action'] = np.where( (((df_temp['close']<df_temp['bb_low'])|(df_temp['close']<df_temp['kc_low']))\
                                &(df_temp['stoch_rsi']<stoch_rsi_low)\
                                &(df_temp['di_down']>di_down_bound)\
                                &(df_temp['adx']>adx_bound)\
                                & (df_temp['di_down']>df_temp['di_up']))\
                                | ((df_temp['close']<df_temp['bb_low'])\
                                & (df_temp['low']<df_temp['s1'])),\
                                 -1, df_temp['action'] ) 
                                 
        if (df_temp["action"]==0).all():
            return df_temp["action"]
        else:
            df_temp['action'] = df_temp['action'].replace(0,np.nan)
            df_temp['action'] = df_temp['action'].fillna(method='ffill').shift(1)

        return df_temp["action"]

class SignalForceStrategy(BaseStrategy):
    
    def set_constants(self,**kwargs):
        for prices in self.time_serie_vec:
            vwap=(prices["close"]*prices["volume"]).rolling(30,min_periods=0).mean()/prices["volume"].rolling(30,min_periods=0).mean()
            sma=(prices["close"]).rolling(30,min_periods=0).mean()
            bbh=bollinger_hband(prices["close"],fillna=True)
            bbl=bollinger_lband(prices["close"],fillna=True)
            prices["MACD"]=macd_diff(prices["close"],fillna=True)
            prices["BB"]=(prices["close"]-bbl)/(bbh-bbl)
            prices.loc[prices.index[0],"BB"] = 0.5
            prices["RSI"]=rsi(prices["close"],fillna=True)
            prices["VOL"]=((vwap/sma)-2*(vwap/sma).rolling(30).std())/(2*(vwap/sma).rolling(30).std())
            prices["MACD"]=(prices["MACD"]-2*prices["MACD"].rolling(26).std())/(4*prices["MACD"].rolling(26).std())
            #prices["VOL"]=10000*prices["VOL"].apply(lambda x : np.log(x))
            prices["VOL_diff"]=prices["VOL"].pct_change()
            prices["MACD_diff"]=prices["MACD"].pct_change()
            prices["BB_diff"]=prices["BB"].pct_change()
            prices["RSI_diff"]=prices["RSI"].pct_change()
            prices.loc[prices.index[0],"VOL_diff"] = 0
            prices.loc[prices.index[0],"MACD_diff"] = 0
            prices.loc[prices.index[0],"BB_diff"] = 0
            prices.loc[np.isnan(prices["RSI_diff"]),"RSI_diff"] = 0
            
        return
    
    def get_strategy_name(self):
        return "signal_force"

    def get_default_parameters_bounds(self):
        return {"macd_requp":(-3,3),
                "macd_reqdown":(-3,3),
                "macd_signalup":(-50,50),
                "macd_signaldown":(-50,50),
                #"macd_diff_requp":(),
                #"macd_diff_reqdown":(),
                "macd_diff_signalup":(-50,50),
                "macd_diff_signaldown":(-50,50),
                "vol_requp":(-3,3),
                "vol_reqdown":(-3,3),
                "vol_signalup":(-50,50),
                "vol_signaldown":(-50,50),
                #"vol_diff_requp":(),
                #"vol_diff_reqdown":(),
                "vol_diff_signalup":(-50,50),
                "vol_diff_signaldown":(-50,50),
                "bb_requp":(0.5,1.5),
                "bb_reqdown":(-0.5,0.5),
                "bb_signalup":(-50,50),
                "bb_signaldown":(-50,50),
                #"bb_diff_requp":(),
                #"bb_diff_reqdown":(),
                "bb_diff_signalup":(-50,50),
                "bb_diff_signaldown":(-50,50),
                "rsi_requp":(50,90),
                "rsi_reqdown":(10,50),
                "rsi_signalup":(-50,50),
                "rsi_signaldown":(-50,50),
                #"rsi_diff_requp":(0),
                #"rsi_diff_reqdown":(0),
                "rsi_diff_signalup":(-50,50),
                "rsi_diff_signaldown":(-50,50)
                }
    
    def get_signal(self, **kwargs):
        macd_requp=kwargs["macd_requp"]
        macd_reqdown=kwargs["macd_reqdown"]
        macd_signalup=kwargs["macd_signalup"]
        macd_signaldown=kwargs["macd_signaldown"]
        macd_diff_signalup=kwargs["macd_diff_signalup"]
        macd_diff_signaldown=kwargs["macd_diff_signaldown"]
        vol_requp=kwargs["vol_requp"]
        vol_reqdown=kwargs["vol_reqdown"]
        vol_signalup=kwargs["vol_signalup"]
        vol_signaldown=kwargs["vol_signaldown"]
        vol_diff_signalup=kwargs["vol_diff_signalup"]
        vol_diff_signaldown=kwargs["vol_diff_signaldown"]
        bb_requp=kwargs["bb_requp"]
        bb_reqdown=kwargs["bb_reqdown"]
        bb_signalup=kwargs["bb_signalup"]
        bb_signaldown=kwargs["bb_signaldown"]
        bb_diff_signalup=kwargs["bb_diff_signalup"]
        bb_diff_signaldown=kwargs["bb_diff_signaldown"]
        rsi_requp=kwargs["rsi_requp"]
        rsi_reqdown=kwargs["rsi_reqdown"]
        rsi_signalup=kwargs["rsi_signalup"]
        rsi_signaldown=kwargs["rsi_signaldown"]
        rsi_diff_signalup=kwargs["rsi_diff_signalup"]
        rsi_diff_signaldown=kwargs["rsi_diff_signaldown"]
        macd_diff_requp,macd_diff_reqdown=0,0
        vol_diff_requp,vol_diff_reqdown=0,0
        bb_diff_requp,bb_diff_reqdown=0,0
        rsi_diff_requp,rsi_diff_reqdown=0,0
        
        prices = self.time_serie
        
        macd_requp=0.5+macd_requp*prices["MACD"].rolling(26).std().values
        macd_reqdown=0.5-macd_reqdown*prices["MACD"].rolling(26).std().values
        vol_requp=0.5+vol_requp*prices["VOL"].rolling(30).std().values
        vol_reqdown=0.5-vol_reqdown*prices["VOL"].rolling(30).std().values
        
        signal=np.zeros(len(prices["open"]))

        MACD=prices["MACD"].values
        VOL=prices["VOL"].values
        RSI=prices["RSI"].values
        BB=prices["BB"].values
        MACD_diff=prices["MACD_diff"].values
        VOL_diff=prices["VOL_diff"].values
        RSI_diff=prices["RSI_diff"].values
        BB_diff=prices["BB_diff"].values
        
        signal=np.zeros(len(prices["open"]))
        #requiments over technical indicators
        
        signal=np.where(MACD>macd_requp,signal+macd_signalup,
                        np.where(MACD<macd_reqdown,signal+macd_signaldown,signal))
                        
        signal=np.where(VOL>vol_requp,signal+vol_signalup,
                        np.where(VOL<vol_reqdown,signal+vol_signaldown,signal))
        
        signal=np.where(RSI>rsi_requp,signal+rsi_signalup,
                        np.where(RSI<rsi_reqdown,signal+rsi_signaldown,signal))
        
        signal=np.where(BB>bb_requp,signal+bb_signalup,
                        np.where(BB<bb_reqdown,signal+bb_signaldown,signal))
        #requiments over diff
        signal=np.where(MACD_diff>macd_diff_requp,signal+macd_diff_signalup,
                        np.where(MACD_diff<macd_diff_reqdown,signal+macd_diff_signaldown,signal))
                        
        signal=np.where(VOL_diff>vol_diff_requp,signal+vol_diff_signalup,
                        np.where(VOL_diff<vol_diff_reqdown,signal+vol_diff_signaldown,signal))
        
        signal=np.where(RSI_diff<rsi_diff_requp,signal+rsi_diff_signalup,
                        np.where(RSI_diff>rsi_diff_reqdown,signal+rsi_diff_signaldown,signal))
        
        signal=np.where(BB_diff<bb_diff_requp,signal+bb_diff_signalup,
                        np.where(BB_diff>bb_diff_reqdown,signal+bb_diff_signaldown,signal))
        return signal
        
    def get_actions(self, **kwargs):
        signal = self.get_signal(**kwargs)
        #signal requirements
        signal = np.where(signal>50,1,
                        np.where(signal<(-50),-1,np.nan))

        return pd.Series(signal,index=self.time_serie.index).ffill().shift(1).replace(np.nan,0)
    
class SignalForceAltStrategy(SignalForceStrategy):
    
    def get_strategy_name(self):
        return "signal_force_alt"

    def get_default_parameters_bounds(self):
        return {"macd_requp":(0,3),
                "macd_reqdown":(0,3),
                #"macd_diff_requp":(),
                #"macd_diff_reqdown":(),
                "macd_signalupup":(-100,100),
                "macd_signaldownup":(-100,100),
                "macd_signalupdown":(-100,100),
                "macd_signaldowndown":(-100,100),
                "vol_requp":(0,3),
                "vol_reqdown":(0,3),
                #"vol_diff_requp":(),
                #"vol_diff_reqdown":(),
                "vol_signalupup":(-100,100),
                "vol_signaldownup":(-100,100),
                "vol_signalupdown":(-100,100),
                "vol_signaldowndown":(-100,100),
                "bb_requp":(0.5,1.5),
                "bb_reqdown":(-0.5,0.5),
                #"bb_diff_requp":(),
                #"bb_diff_reqdown":(),
                "bb_signalupup":(-100,100),
                "bb_signaldownup":(-100,100),
                "bb_signalupdown":(-100,100),
                "bb_signaldowndown":(-100,100),
                "rsi_requp":(60,90),
                "rsi_reqdown":(10,40),
                #"rsi_diff_requp":(0),
                #"rsi_diff_reqdown":(0),
                "rsi_signalupup":(-100,100),
                "rsi_signalupdown":(-100,100),
                "rsi_signaldownup":(-100,100),
                "rsi_signaldowndown":(-100,100),
                }
    
    def get_signal(self, **kwargs):
        macd_requp=kwargs["macd_requp"]
        macd_reqdown=kwargs["macd_reqdown"]
        macd_signalupup=kwargs["macd_signalupup"]
        macd_signalupdown=kwargs["macd_signalupdown"]
        macd_signaldownup=kwargs["macd_signaldownup"]
        macd_signaldowndown=kwargs["macd_signaldowndown"]
        
        vol_requp=kwargs["vol_requp"]
        vol_reqdown=kwargs["vol_reqdown"]
        vol_signalupup=kwargs["vol_signalupup"]
        vol_signalupdown=kwargs["vol_signalupdown"]
        vol_signaldownup=kwargs["vol_signaldownup"]
        vol_signaldowndown=kwargs["vol_signaldowndown"]
        
        bb_requp=kwargs["bb_requp"]
        bb_reqdown=kwargs["bb_reqdown"]
        bb_signalupup=kwargs["bb_signalupup"]
        bb_signalupdown=kwargs["bb_signalupdown"]
        bb_signaldownup=kwargs["bb_signaldownup"]
        bb_signaldowndown=kwargs["bb_signaldowndown"]
        
        rsi_requp=kwargs["rsi_requp"]
        rsi_reqdown=kwargs["rsi_reqdown"]
        rsi_signalupup=kwargs["rsi_signalupup"]
        rsi_signalupdown=kwargs["rsi_signalupdown"]
        rsi_signaldownup=kwargs["rsi_signaldownup"]
        rsi_signaldowndown=kwargs["rsi_signaldowndown"]
        
        macd_diff_requp,macd_diff_reqdown=0,0
        vol_diff_requp,vol_diff_reqdown=0,0
        bb_diff_requp,bb_diff_reqdown=0,0
        rsi_diff_requp,rsi_diff_reqdown=0,0
                
        prices=self.time_serie.copy()        
        
        macd_requp=0.5+macd_requp*prices["MACD"].rolling(26).std().values
        macd_reqdown=0.5-macd_reqdown*prices["MACD"].rolling(26).std().values
        vol_requp=0.5+vol_requp*prices["VOL"].rolling(30).std().values
        vol_reqdown=0.5-vol_reqdown*prices["VOL"].rolling(30).std().values        
        
        MACD=prices["MACD"].values
        VOL=prices["VOL"].values
        RSI=prices["RSI"].values
        BB=prices["BB"].values
        MACD_diff=prices["MACD_diff"].values
        VOL_diff=prices["VOL_diff"].values
        RSI_diff=prices["RSI_diff"].values
        BB_diff=prices["BB_diff"].values
        
        signal=np.zeros(len(prices["open"]))
        #requiments over technical indicators
        signal[(MACD>macd_requp) & (MACD_diff>macd_diff_requp)]+=macd_signalupup
        signal[(MACD>macd_requp) & (MACD_diff<macd_diff_reqdown)]+=macd_signalupdown
        signal[(MACD<macd_reqdown) & (MACD_diff>macd_diff_requp)]+=macd_signaldownup
        signal[(MACD<macd_reqdown) & (MACD_diff<macd_diff_reqdown)]+=macd_signaldowndown
        
        signal[(VOL>vol_requp) & (VOL_diff>vol_diff_requp)]+=vol_signalupup
        signal[(VOL>vol_requp) & (VOL_diff<vol_diff_reqdown)]+=vol_signalupdown
        signal[(VOL<vol_reqdown) & (VOL_diff>vol_diff_requp)]+=vol_signaldownup
        signal[(VOL<vol_reqdown) & (VOL_diff<vol_diff_reqdown)]+=vol_signaldowndown
        
        signal[(BB>bb_requp) & (BB_diff>bb_diff_requp)]+=bb_signalupup
        signal[(BB>bb_requp) & (BB_diff<bb_diff_reqdown)]+=bb_signalupdown
        signal[(BB<bb_reqdown) & (BB_diff>bb_diff_requp)]+=bb_signaldownup
        signal[(BB<bb_reqdown) & (BB_diff<bb_diff_reqdown)]+=bb_signaldowndown
        
        signal[(RSI>rsi_requp) & (RSI_diff>rsi_diff_requp)]+=rsi_signalupup
        signal[(RSI>rsi_requp) & (RSI_diff<rsi_diff_reqdown)]+=rsi_signalupdown
        signal[(RSI<rsi_reqdown) & (RSI_diff>rsi_diff_requp)]+=rsi_signaldownup
        signal[(RSI<rsi_reqdown) & (RSI_diff<rsi_diff_reqdown)]+=rsi_signaldowndown
        
        return signal

##############################################################################

class LongTraillingStopLoss(BaseStrategy):
    
    '''
    AGREGAR:
        Cruce de medias largas para anular trades hasta que de positivo
        si no hay soporte abajo de last, no operar
        Caida fuerte, se va por 5 obs
        Sacar trailing hacia afuera
    '''
    
    
    def get_strategy_name(self):
        return "LongTraillingStopLoss"
    
    def get_default_parameters_bounds(self):
        return {'obs_pps': (5, 15), 
                'mean_atr': (15, 25)
                }
        
    def is_support(self, df, i, n):
        suport_candidate = df.close[i]
        local_min = np.min(df.close[i-n: i+n+1])
        return suport_candidate == local_min

    def is_resistance(self, df, i, n):
        resistance_candidate = df.close[i]
        local_max = np.max(df.close[i-n: i+n+1])
        return resistance_candidate == local_max
    
    def is_far_from_level(self, l, s, levels):
        return np.sum([abs(l-x[2]) < s for x in levels]) == 0
        
    def combine_level(self, l, s, levels):
        obs, rep, prices = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in levels:
              if abs(l-x[2]) < s:
                  obs, rep, prices = np.append(obs, x[0]), np.append(rep, x[1]), np.append(prices, x[2])
        levels = [x for x in levels if x[0] not in obs] #Elimina los duplicados
        levels.append((obs.min(), rep.sum()+1, np.append(prices, l).mean()))
        return levels
                            
    def find_pivot_points(self, df, obs_pps):
        pps = []
        for i in range(obs_pps, df.shape[0]-obs_pps):
            if self.is_support(df,i, obs_pps):
                l = df['close'][i]
                pps.append((df.index[i],l))
            elif self.is_resistance(df,i, obs_pps):
                l = df['close'][i]
                pps.append((df.index[i],l))
        return pps
    
    def find_important_areas(self, df, obs_pps, mean_atr):
        df = self.calculate_average_true_range(df, mean_atr)
        pps = self.find_pivot_points(df, obs_pps)
        df['areas'] = 0
        df['areas'] = df['areas'].astype(object)
        for i in range(len(df)):
            df.at[df.index[i], 'areas'] = self.combine_area_expiration(i, df, pps)
        return df, pps
            
    def combine_area_expiration(self, i, df, pps):
        idx_start, idx_end = df.index[0] if i<100 else df.index[i-100], df.index[i]
        pps_to_consider = [pp for pp in pps if pp[0] >= idx_start and pp[0] <= idx_end]
        
        areas, s = [], df[idx_start: idx_end]['ATR'].mean() * 0.25
        for pp in pps_to_consider:
            if self.is_far_from_level(pp[1], s, areas):
                areas.append((pp[0],0,pp[1],pp[1]))
            else:
                areas = self.combine_area(pp[1], s, areas)
        return areas
    
    def combine_area(self, l, s, areas):
        obs, rep, maxs, mins = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in areas:
              if (abs(l-x[2]) < s or abs(l-x[3]) < s):
                  obs, rep, maxs, mins = np.append(obs, x[0]), np.append(rep, x[1]), np.append(maxs, x[2]), np.append(mins, x[3])
        areas = [x for x in areas if x[0] not in obs] #Elimina los duplicados
        areas.append((obs.min(), rep.sum()+1, np.append(maxs, l).max(), np.append(mins, l).min()))
        return areas
                
    def calculate_true_range(self, df):
        df['tr1'] = df["high"] - df["low"]
        df['tr2'] = abs(df["high"] - df["close"].shift(1))
        df['tr3'] = abs(df["low"] - df["close"].shift(1))
        df['TR'] = df[['tr1','tr2','tr3']].max(axis=1)
        df.loc[df.index[0],'TR'] = 0
        return df
    
    def calculate_average_true_range(self, df, mean):
        df = self.calculate_true_range(df)
        df['ATR'] = 0
        try:
            df.loc[df.index[mean],'ATR'] = round( df.loc[df.index[1:mean+1],"TR"].rolling(window=mean).mean()[mean], 4)
        except:
            df.loc[df.index[mean],'ATR'] = round( df.loc[df.index[1:mean+1],"TR"].rolling(window=mean).mean()[-1], 4)
        const_atr = (mean-1)/mean
        const_tr = 1/mean
        ATR=df["ATR"].values
        TR=df["TR"].values
        for index in range(mean+1, len(df)):
            ATR[index]=ATR[index-1]*const_atr+TR[index]*const_tr
        df["ATR"]=ATR
        return df
    
    def red_cross(self, df):
        df["ema_short"] = df["close"].rolling(50).mean()
        df["ema_long"] = df["close"].rolling(100).mean()
        df["red_cross"] = 0
        df.loc[(df["ema_long"] > df["ema_short"]) ,"red_cross"] = -1
        df.loc[(df["ema_short"] > df["ema_long"]),"red_cross"] = 1
        return df["red_cross"]
    
    def long_trailling_stop(self, df, umbral):
        df['ret'] = df.open.pct_change()
        df.loc[df.action == 0, 'ret'] = 0.
        df['ret_accum'] = 0.
        df['stop_light'] = 0.
        start_index = 0
        for row in range(len(df)):
            if (df.action[row] == 1) and (df.stop_light[row] == 0):
                df['ret_accum'].iloc[row] = df.ret[start_index:row].cumsum()[-1]
                if df['ret_accum'][row] < -umbral:
                    df['stop_light'].iloc[row:row+10] = 1
                    df['action'].iloc[row:row+10] = -1  
                    start_index = row
        return df['action']
    
    def get_actions(self, **kwargs):
        obs_pps = int(kwargs['obs_pps'])
        mean_atr = int(kwargs['mean_atr'])
        
        df = self.time_serie.copy()   
        df, _ = self.find_important_areas(df, obs_pps, mean_atr) 
        '''
        Se vende si:
            1) No se está dentro de una zona pivote, [condicion de lateralidad]
            2) Se tiene al menos 1 ATRs de recorrido al alza hasta la zona pivote superior más cercana. [condicion de recorrido alcista]
            3) Si el cruce de medias largas da venta
        '''
        df['action'] = 0
        df['condition1'],df['condition2'],df['condition3'] = 0,1,0
        for row in range(1,len(df)):
            df.condition1[row] = 1 not in [1 for area in df.areas[row] if (area[2] >= df.close[row] >= area[3] and df.index[row]>area[0])]                          
            try:
                df.condition2[row] = (( 1/2*df.ATR[row] < np.min([df.close[row]-area[2] for area in df.areas[row] if (df.close[row]-area[2]>0 and area[1]>0)]) ) )
            except:
                df.condition2[row] = True
        df['condition3'] = self.red_cross(df)
        df['action'] = np.where((df['condition1']==1) &
                                (df['condition2']==1) &
                                (df['condition3']==1),
                                1, -1)
        #umbral = 0.05
        #df['action'] = self.long_trailling_stop(df, umbral)
        
        if (df["action"]==0).all():
            return df["action"]
        else:
            df['action'] = df['action'].replace(0,np.nan)
            df['action'] = df['action'].fillna(method='ffill').shift(1)
        return df["action"]
    
    
class BBvolumeStrategy(BaseStrategy):
    def set_constants(self,**kwargs):
        return
  
    def get_strategy_name(self):
        return "BBvolume"

    def get_default_parameters_bounds(self):
        return {"bb_mult_factor":(1.5,3),
                "bb_periods":(20,50),
                "vol_periods":(1*(self.periods/365),7*(self.periods/365)),
                "vol_mult_factor_up":(1,3),
                "vol_mult_factor_down":(1,3),
#                "pct_candle":(2,50),
                
                }

    def get_actions(self, **kwargs):
        bb_mult_factor = kwargs["bb_mult_factor"]
        bb_lenght = int(kwargs["bb_periods"])
        vol_lenght = int(kwargs["vol_periods"])
        vol_mult_factor_up = kwargs["vol_mult_factor_up"]
        vol_mult_factor_down = kwargs["vol_mult_factor_down"]
        
   #     pct_candle = kwargs["pct_candle"]
       # stop_loss = kwargs["stop_loss"]
        #take_profit = kwargs["take_profit"]
        
        price=self.time_serie    
        price["sma"] = price["close"].rolling(bb_lenght).mean()
        
        price["bbu"] = price["close"].rolling(bb_lenght).mean() + \
            bb_mult_factor*price["close"].rolling(bb_lenght).std()
        price["bbl"] = price["close"].rolling(bb_lenght).mean() - \
            bb_mult_factor*price["close"].rolling(bb_lenght).std()
        price["kcu"] = keltner_channel_hband(price["high"],price["low"],price["close"],n=bb_lenght)
        price["kcl"] = keltner_channel_lband(price["high"],price["low"],price["close"],n=bb_lenght)
        
        price["vol_change"] = price["volume"].rolling(vol_lenght).mean()
        
        price["action"] = np.nan
        price["action"] = np.where((price["bbu"]<price["close"]) & \
             (price["kcu"]<price["close"]) & \
             ((price["bbu"]>price["close"].shift(1)) |\
             (price["kcu"]>price["close"].shift(1))) &\
             (price["volume"]>vol_mult_factor_up*price["vol_change"])\
             , 2,price["action"])
        price["action"] = np.where((price["bbl"]>price["close"]) & \
             (price["kcl"]>price["close"]) & \
             ((price["bbl"]<price["close"].shift(1)) |\
             (price["kcl"]<price["close"].shift(1))) &\
             (price["volume"]<vol_mult_factor_down*price["vol_change"])\
             ,1, price["action"])
        price["action"] = np.where((price["bbu"]<price["close"]) &\
             (price["kcu"]<price["close"]) &\
             ((price["bbu"]>price["close"].shift(1)) |\
             (price["kcu"]>price["close"].shift(1))) &\
             (price["volume"]<vol_mult_factor_up*price["vol_change"])\
             ,-1, price["action"])
        price["action"] = np.where((price["bbl"]>price["close"]) &\
             (price["kcl"]>price["close"]) &\
             ((price["bbl"]<price["close"].shift(1)) |\
             (price["kcl"]<price["close"].shift(1))) &\
             (price["volume"]>vol_mult_factor_down*price["vol_change"])\
             , -2, price["action"])
       
        status = price["action"].ffill()
        price["action"] = np.where((status==2) &\
             (price["sma"]>=price["close"])
             ,0, price["action"])
        price["action"] = np.where((status==-2) &\
             (price["sma"]<=price["close"])
             ,0, price["action"])
         
        
#        action = price.action.values
#        pct = price.close.pct_change().values
#        counting = False
#        short=False
#        cumprod = 1
#        
#        for i in range(len(price)):
#            
#            if action[i] == 1:
#                counting = True
#                short = False
#                cumprod = 1
#            elif action[i] == -1:
#                counting = True
#                short = True
#                cumprod = 1
#            
#            if cumprod>take_profit:
#                counting = False
#                cumprod = 1
#                action[i] = 0
#            elif cumprod<stop_loss:
#                counting = False
#                cumprod = 1
#                action[i]=0
#                
#            if counting:
#                if short:
#                    cumprod *= (-1)*pct[i]
#                    action[i] = -1
#                else:
#                    cumprod *= pct[i]
#                    action[i] = 1
#        price["action"] = action
#
        price["action"]=price["action"].replace(2,1).replace(-2,-1)
        price["action"]=price["action"].shift(1).ffill().replace(np.nan,0)
        return price["action"]
