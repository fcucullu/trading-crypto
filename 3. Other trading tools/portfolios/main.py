import pandas as pd
from datetime import datetime
from datetime import timedelta

from src.HyperParametersGenerator import BayesianHyperParametersGenerator
from src.ModelOptimization import ModelOptimization
from src.Function_Portfolio import Recomendador
from src.Function_Portfolio import ObjectiveFunction
import quantstats as qs
#from fillna import fillna

def main(coins_to_consider,quote, train_weeks,volatility=100,n_iter=1000,
         init_points=100,optimize="sharpe",
         compression="2h",last_date="last",
         set_pendiente_to_zero=False):
    
    if train_weeks<=0:
        print("train_weeks must be greater than 0")

    one_week={"1w":1,
              "1d":7,
            "12h":2*7,
            "6h":4*7,
            "2h":12*7,
            "1h":24*7,
            "30m":48*7,
            "15m":96*7}
    periods=one_week[compression]*(365/7)

    #Levanto y limpio los precios para utilizarcoins_to_consider=["BTC","BNB","ETH", 'LINK','USDT']
    prices=Recomendador(1,1,compression,"USDT",1,1).get_df(coins_to_consider)[int(quote=="BTC")]
    prices=prices.dropna()
    prices=prices.asfreq(compression.replace("m","t")).ffill()
    if last_date=="last":
        date_final = prices.index[-1]
    else:
        date_final = prices.loc[datetime.strptime(last_date,"%Y-%m-%d")].name
    date_init = max(date_final-timedelta(days=int(7*train_weeks)),\
                    prices.index[100])
    #Defino y seteo la función objetivo
    
    ob_func=ObjectiveFunction(prices,date_init,date_final,output=optimize,periods=periods)
    ob_func.set_maximum_volatility(volatility)
    ob_func.init_df_history()
    #ob_func.set_plot(True)
    
    #defino el optimizador
    opt=ModelOptimization(ob_func,
                          BayesianHyperParametersGenerator
                  )
    #aplico este método que me devuelve un DataFrame con los resultados test y train para un conjunto de parámetros
    
    bounds={"riesgo":(0,100),
            "lecturas":(2,100),
            "pendiente":(0,5000),
            "factorcom":(0,20)}
    if set_pendiente_to_zero:
        bounds["pendiente"]=(0,0)

    result=opt.train_model(bounds,
            prices.iloc[-train_weeks*one_week[compression]:]
            ,n_iter=n_iter,
            init_points=init_points,
            verbose=2)
    
    
    optimal=pd.Series(result[0])
    
    returns=ob_func.get_returns(**result[0])
    expected_return=((1+returns).cumprod()[-1])**(periods/len(returns))-1
    volatility=qs.stats.volatility(returns,periods=periods)
    
    optimal["expected_return"]=expected_return
    optimal["sharpe"]=expected_return/volatility
    optimal["volatility"]=volatility
    optimal["kelly"]=qs.stats.kelly_criterion(returns)
    optimal["date_final"]=date_final
    
    return optimal
