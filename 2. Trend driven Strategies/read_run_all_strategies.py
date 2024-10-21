#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:19:41 2020

@author: farduh
"""
from src.OptimizeStrategyParameters import OptimizeStrategyParameters
from src.Estrategia_por_par import *
from src.PriceReader import PriceReader
import pandas as pd
import warnings
from datetime import timedelta 
from datetime import datetime
import itertools
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/farduh/xcapit/xcapit_util")
from xcapit_metrics import Xmetrics
from recommendations import BrainyRecommendationsRepository
from recommendations import PRO_BTC, CLASSIC_BTC, PRO_USDT, CLASSIC_USDT
from recommendations_runner import RecommendationsRunner, create_full_recommendation
from prices import PricesRepository
import pandas as pd

strategies={"est_pps":EstrategiaPivotPoints,
             "est_2sma":EstrategiaDosMedias,
             "est_2sma_lat":EstrategiaDosMediasLateral,
             "est_3sma":EstrategiaTresMedias,
             "est_3sma_lat":EstrategiaTresMediasLateral,
             "est_3sma2":EstrategiaTresMedias2,
             "est_3sma2_lat":EstrategiaTresMedias2Lateral,
             "est_vsma":EstrategiaVWAPvsSMA,
             "est_vsma_lat":EstrategiaVWAPvsSMALateral,
             #"est_pend":EstrategiaPendiente.
             }


pbounds_margin={"mult_lateral":(.5,2),
         "margin_short":(0,3),
         "margin_long":(0,3),
         "porc_to_inv":(0,1)}


pbounds={"est_2sma":{"short":(1,50),
                    "long":(10,400)},
        "est_2sma_lat":{"short":(1,50),
                    "long":(10,400),
                    "mult_lateral":(.5,2)},                    
        "est_3sma":{"short":(1,50),
                    "medium":(10,200),
                    "long":(30,400)},
        "est_3sma_lat":{"short":(1,50),
                    "medium":(10,200),
                    "long":(30,400),
                    "mult_lateral":(.5,2)},
        "est_3sma2":{"short":(1,50),
                    "medium":(10,200),
                    "long":(30,400),
                    },
         "est_3sma2_lat":{"short":(1,50),
                    "medium":(10,200),
                    "long":(30,400),
                    "mult_lateral":(.5,2),
                    },
         "est_vsma":{"lecturas":(5,2000),
                     "porcentaje":(0,0.5),
                     },
         "est_vsma_lat":{"lecturas":(5,2000),
                         "porcentaje":(0,0.5),
                         "mult_lateral":(.5,2),
                         },
         
        "est_pend":{"short":(1,40),
                    "medium":(10,200),
                    "long":(30,400),
                    },
         "est_pps":{}       
        }

for key in strategies.keys():
    pbounds[key].update(pbounds_margin)


comision_margin_per_day={
    "USDT":(0.05e-2),
    "BTC":(0.02e-2),
    "BNB":(0.3e-2),
    "LTC":(0.02e-2),
    "ETH":(0.0275e-2),
    "BCH":(0.02e-2),
    "ETC":(0.02e-2),
    "XRP":(0.01e-2),
    "EOS":(0.02e-2),
    }

comision_margin=pd.Series(comision_margin_per_day)
comision_margin=(1+comision_margin)**(1/12)-1


sopt=OptimizeStrategyParameters()

dates=[datetime(2020,6,1)-timedelta(days=7*x) for x in range(1,14)]


strategies={"est_pps":EstrategiaPivotPoints,
             "est_2sma":EstrategiaDosMedias,
             "est_2sma_lat":EstrategiaDosMediasLateral,
             "est_3sma":EstrategiaTresMedias,
             "est_3sma_lat":EstrategiaTresMediasLateral,
             "est_3sma2":EstrategiaTresMedias2,
             "est_3sma2_lat":EstrategiaTresMedias2Lateral,
 #            "est_vsma":EstrategiaVWAPvsSMA,
 #            "est_vsma_lat":EstrategiaVWAPvsSMALateral,
             #"est_pend":EstrategiaPendiente
             }

##
## Optimización con simulaciones
##

data_types=["historical","simulated"]
compressions=["15t","1h"]
metrics_to_opt=["sharpe","kelly","return"]


### mergeo con las corridas locales
file_tag="2020-06-19_20"
optimal=pd.read_csv(f"tests/optimal_parameters_{file_tag}.csv")
optimal_to_merge=pd.read_csv(f"tests/optimal_parameters_{file_tag}_to_merge.csv")
optimal=optimal.append(optimal_to_merge,ignore_index=True)
optimal.to_csv("tests/optimal_parameters.csv",index=False)

#optimal[optimal[["compression","data_type","date_final","estrategia","optimize"]].duplicated()]

### chequeo de puntos faltantes

optimal=pd.read_csv("tests/optimal_parameters.csv")
optimal["date_final"]=pd.to_datetime(optimal["date_final"])

for compression,data_type,key,date,metric in itertools.product(compressions,data_types,strategies.keys(),sorted(dates),metrics_to_opt):
    if optimal[(optimal["date_final"]==date)&
               (optimal["estrategia"]== key)&
               (optimal["compression"]==compression)&
               (optimal["data_type"]==data_type) &
               (optimal["optimize"]==metric)
               ].empty:
        print(f'("{data_type}",datetime({date.strftime("%Y,%m,%d")}),"{compression}","{key}","{metric}")')
    
##
## Leo y calculo los retornos con los parámetros óptimos optenidos por simulaciones y datos
##

 
#returns.to_csv("tests/est_ret.csv")



recommendations = BrainyRecommendationsRepository().get_recommendations(PRO_BTC)

recommendations = create_full_recommendation("BTC", recommendations[["BTC","ETH","BNB","LTC","USDT"]])
prices_btc = PricesRepository().get_prices('BTC', datetime(2020,2,20), datetime(2020,6,2),n_minutes = 120,candlestick_label = 'close').ffill()
portfolio, returns_pro = RecommendationsRunner().run('BTC', recommendations, prices_btc)



##
## Grafico las estrategias
##
            
##Graficos obtimizando sharpe

##comparing compressions
for metric in metrics_to_opt:
    
    plt.figure(figsize=(40,20))
    plt.title(f"metrica {metric}")
    for compression in compressions:
        l=0
        price=(pr.get_price("USDTBTC",compression).dropna().asfreq(compression).ffill())
        for key in strategies.keys():
            l+=1
            plt.subplot(4,2,l)
            plt.title(f"{key}")
            if compression=="15t":
                (price["close"].loc[returns_dict[compression].index]/price["close"].loc[returns_dict[compression].index[0]]).plot(label="USDTBTC")
                (1+returns_pro.loc[returns_dict[compression].index[0]:returns_dict[compression].index[-1]]).cumprod().plot(label="pro_btc",linestyle="--",c="black")
            (1+returns_dict[compression][f"{key}_hist_{metric}"]).cumprod().plot(label=f"data_{compression}")
            (1+returns_dict[compression][f"{key}_sim_{metric}"]).cumprod().plot(label=f"sim_{compression}")
    
    
    
            plt.legend()
    plt.savefig(f"plots/strategies_{metric}.pdf")

##comparing metric to optimize
for compression in compressions:
    
    plt.figure(figsize=(40,20))
    plt.title(f"compresion {compression}")
    for metric in metrics_to_opt:
        l=0
        price=(pr.get_price("USDTBTC",compression).dropna().asfreq(compression).ffill())
        for key in strategies.keys():
            l+=1
            plt.subplot(4,2,l)
            plt.title(f"{key}")
            if metric=="sharpe":
                (price["close"].loc[returns_dict[compression].index]/price["close"].loc[returns_dict[compression].index[0]]).plot(label="USDTBTC")
                (1+returns_pro.loc[returns_dict[compression].index[0]:returns_dict[compression].index[-1]]).cumprod().plot(label="pro_btc",linestyle="--",c="black")
            (1+returns_dict[compression][f"{key}_hist_{metric}"]).cumprod().plot(label=f"data_{metric}")
            (1+returns_dict[compression][f"{key}_sim_{metric}"]).cumprod().plot(label=f"sim_{metric}")
            
            plt.legend()
    plt.savefig(f"plots/strategies_{compression}.pdf")



##
## Xcapit ratio
##




##
## Cálculo de métricas
##


m=Xmetrics()

pd.options.display.max_columns=20
pd.set_option('display.width', 1000)

df_days=pd.DataFrame()
df_inv=pd.DataFrame()

df_days=df_days.append(m._calculate_days_metrics( f"pro_btc",returns_pro.loc[datetime(2020,3,1):datetime(2020,6,1)],m.PRO_BTC_THRESHOLDS))
df_inv=df_inv.append(m.get_investor_metrics(f"pro_btc",returns_pro.loc[datetime(2020,3,1):datetime(2020,6,1)],365*12))

for compression,returns in returns_dict.items():
    prices=(pr.get_price("USDTBTC",compression).dropna().asfreq(compression).ffill())
    
    df_days=df_days.append(m._calculate_days_metrics( f"USDTBTC_{compression}",prices.close.pct_change().loc[datetime(2020,3,1):datetime(2020,6,1)],m.PRO_BTC_THRESHOLDS))
    df_inv=df_inv.append(m.get_investor_metrics(f"USDTBTC_{compression}",prices.close.pct_change().loc[datetime(2020,3,1):datetime(2020,6,1)],periods[compression]))
    
    
    
    for col in returns.columns:
        
        df_days=df_days.append(m._calculate_days_metrics( f"{col}_{compression}",returns[col].loc[datetime(2020,3,1):datetime(2020,6,1)],m.PRO_BTC_THRESHOLDS))
        df_inv=df_inv.append(m.get_investor_metrics(f"{col}_{compression}",returns[col].loc[datetime(2020,3,1):datetime(2020,6,1)],periods[compression]))
        #df_days=df_days.append(m._calculate_days_metrics( f"{col}_{compression}",returns[col].loc[datetime(2020,3,1):datetime(2020,6,1)],m.PRO_BTC_THRESHOLDS))
        #df_inv=df_inv.append(m.get_investor_metrics(f"{col}_{compression}",returns[col].loc[datetime(2020,3,1):datetime(2020,6,1)],periods[compression]))
    
        #df_days=df_days.append(m._calculate_days_metrics( "pro_btc",pro_btc_portfolio_evolution.pct_change().replace(np.nan,0),m.PRO_BTC_THRESHOLDS))
        #df_inv=df_inv.append(m.get_investor_metrics("pro_btc",pro_btc_portfolio_evolution.pct_change().replace(np.nan,0),periods[compression]))
   # print(df_inv.T)
    #print(df_days.T)
    
df_inv.to_csv("tests/metrics_inv.csv")
df_days.to_csv("tests/metrics_days.csv")
    
df_inv=df_inv.reset_index()
df_days=df_days.reset_index()    
    