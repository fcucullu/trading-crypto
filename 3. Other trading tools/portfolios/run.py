import ray
import itertools
from main import main
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from src.candlestick import CandlestickRepository

#portfolios = [["BTC","BNB","ETH", 'LINK','USDT'],["BTC","BNB","ETH", 'LTC','USDT']]
portfolios = [["BTC","BNB","ETH", 'LINK','USDT','LTC','LINKDOWN'],
              ["BTC","BNB","ETH", 'LINK','USDT','LTC','LINKDOWN','BTCDOWN'],
              ["BTC","BNB","ETH", 'LINK','USDT'],
              ["BTC","BNB","ETH", 'LINK','USDT','LTC','LINKUP','LINKDOWN','BTCUP','BTCDOWN'],
              ['BTCUP','BTCDOWN','BTC','USDT'],
              ['LINKUP','LINKDOWN' ,'BTC','USDT'],
              ['LINKUP','LINKDOWN','BTCUP','BTCDOWN','BTC','USDT'],
              ["BTC","BNB","ETH", 'LTC', 'USDT'],
              ["BTC","BNB","ETH", 'LINK', 'USDT','LTC'],
              ]


quotes = ["BTC","USDT"]
compressions = ["2h"]
pendiente0 = [True]
init_points={True:400,False:800}
n_iter={True:500,False:600}
dates=["last"]

#portfolios = [["BTC","BNB","ETH", 'LTC','USDT']]
#quotes = ["USDT","BTC"]
#compressions = ["1h","2h","6h","12h","1d"]
#dates = [datetime(2020,7,31)]
# l=1
# while dates[-1]<datetime(2020,11,18):
#     dates.append(dates[0]+timedelta(weeks=(2*l)))
#     l+=1
# dates=dates[:-1]
# dates=[date.strftime("%Y-%m-%d") for date in dates]


repo = CandlestickRepository.preprod_repository()

bases = []
for portfolio in portfolios:
    bases += portfolio
bases = set(bases)

for base,compression in itertools.product(bases,compressions):
    for quote in quotes:
        if (quote == base) or (quote=="BTC" and base=="USDT") \
        or (quote == "BTC" and ("DOWN" in base or "UP" in base)):
            continue
        ticker=f"{base}/{quote}"
        sample = repo.get_candlestick(ticker,'binance',120,datetime(2019,1,1),datetime.now())
        if ("DOWN" in base) or ("UP" in base):
            if ("DOWN" in base):
                base_temp=base[:-4]
            else :
                base_temp=base[:-2]
                
            ticker=f"{base_temp}/{quote}"
            sample_temp = repo.get_candlestick(ticker,'binance',120,datetime(2019,1,1),datetime.now())
            first_nonan = sample.dropna().index[0]
            for col in  ["open","close"]:
                simulation =(1+sample_temp[col].pct_change()*2.5).cumprod()
                sample.loc[:first_nonan,col]=(sample.loc[first_nonan,col]/simulation.loc[first_nonan])*simulation.loc[:first_nonan]
        
        
        sample.to_csv(f"data/{base}{quote}_{compression}.csv")


ray.init(webui_host='0.0.0.0')#,num_cpus=1)

@ray.remote
def markowitz_optimization(quote: str,bases,compression,date,set_pendiente_to_zero):

    string_bases = ""
    for base in sorted(bases):
        string_bases += base+"-"
    string_bases = string_bases[:-1]

    try:
        print(f"tests/{date}_00_optimal_parameters_{string_bases}_quote_{quote.upper()}_{compression}_pendiente0.csv")
        pd.read_csv(f"tests/{date}_00_optimal_parameters_{string_bases}_quote_{quote.upper()}_{compression}_pendiente0.csv")
        print('Already optimized')
        return
    except:
        pass
    print("paso "+string_bases)
    optimal = pd.DataFrame()
    optimal[f'pro_{quote}'] = main(coins_to_consider=bases,
                                       quote=quote, 
                                       train_weeks=30,
                                       n_iter=n_iter[set_pendiente_to_zero],
                                       init_points=init_points[set_pendiente_to_zero],
                                       optimize="sharpe",
                                       compression=compression,
                                       last_date=date,
                                       set_pendiente_to_zero=set_pendiente_to_zero)
    if set_pendiente_to_zero:
        optimal.to_csv(
                "tests/"+optimal.loc["date_final"].iloc[0].strftime("%Y-%m-%d_%H")+f"_optimal_parameters_{string_bases}_quote_{quote.upper()}_{compression}_pendiente0.csv")
    else:
        optimal.to_csv(
                "tests/"+optimal.loc["date_final"].iloc[0].strftime("%Y-%m-%d_%H")+f"_optimal_parameters_{string_bases}_quote_{quote.upper()}_{compression}.csv")
        
futures = [markowitz_optimization.remote(quote,bases,compression,date,set_pendiente_to_zero)\
           for bases,date,quote,compression,set_pendiente_to_zero in\
           itertools.product(portfolios,dates,quotes,compressions,pendiente0)]
          
ray.get(futures)



