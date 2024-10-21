# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/farduh/xcapit/xcapit_util")

from candlestick import CandlestickRepository
from datetime import datetime
import os
import pandas as pd


def download_data(path="/home/farduh/xcapit/strategy_optimizer/samples",
                  bases=["ETC","BTC","ETH","BNB","LTC","BCH","EOS","XRP","USDT"],
                  quotes=["USDT","BTC"]):
    
    repo = CandlestickRepository.preprod_repository()
    
    bases=["USDT","ETH","BNB","LTC","BTC"]    
    quotes=["BTC","USDT"]
    
    
    def BTCUSDT_to_USDTBTC(data):
        
        data["close"]=1000/data["close"]
        data["open"]=1000/data["open"]
        low=1000/data["high"]
        high=1000/data["low"]
        data["high"]=high
        data["low"]=low
        
        return data
    
    for base in bases:
        for quote in quotes:
            if quote==base:
                continue
            print(base+quote)
            if (quote=="BTC") and (base=="USDT"):
                
                data = repo.get_candlestick(f'{quote}/{base}',  'binance', 15, datetime(2018,12,1), datetime.now())
                data=BTCUSDT_to_USDTBTC(data)
                data.index += pd.Timedelta(15, unit="t")-pd.Timedelta(1, unit="ms")
                data.to_csv(f"{path}/{base}{quote}_15t.csv",index_label="close_time")
                
                data = repo.get_candlestick(f'{quote}/{base}',  'binance', 60, datetime(2018,12,1), datetime.now())
                data=BTCUSDT_to_USDTBTC(data)
                data.index += pd.Timedelta(1, unit="h")-pd.Timedelta(1, unit="ms")
                data.to_csv(f"{path}/{base}{quote}_1h.csv",index_label="close_time")
                
                
                data = repo.get_candlestick(f'{quote}/{base}',  'binance', 120, datetime(2018,12,1), datetime.now())
                data=BTCUSDT_to_USDTBTC(data)
                data.index += pd.Timedelta(2, unit="h")-pd.Timedelta(1, unit="ms")
                data.to_csv(f"{path}/{base}{quote}_2h.csv",index_label="close_time")
                
                
                data = repo.get_candlestick(f'{quote}/{base}',  'binance', 1440, datetime(2018,12,1), datetime.now())
                data=BTCUSDT_to_USDTBTC(data)
                data.index += pd.Timedelta(1, unit="d")-pd.Timedelta(1, unit="ms")
                data.to_csv(f"{path}/{base}{quote}_1d.csv",index_label="close_time")
    
                continue
            
            
            data = repo.get_candlestick(f'{base}/{quote}',  'binance', 120, datetime(2018,12,1), datetime.now())
            data.index += pd.Timedelta(2, unit="h")-pd.Timedelta(1, unit="ms")
            if data.empty:
                print(f'{base}/{quote} can not be downloaded')
                exit(0)
            if base=="BCH": data=joinBCHfile(path,data,quote,"2h")
            data.to_csv(f"{path}/{base}{quote}_2h.csv",index_label="close_time")
           
            
            data = repo.get_candlestick(f'{base}/{quote}',  'binance', 1440, datetime(2018,12,1), datetime.now())
            data.index += pd.Timedelta(1, unit="d")-pd.Timedelta(1, unit="ms")
            data=data.reset_index().rename(columns={"index":"close_time"})
            if base=="BCH": data=joinBCHfile(path,data,quote,"1d")
            data.to_csv(f"{path}/{base}{quote}_1d.csv",index_label="close_time")
    
    
    
def joinBCHfile(path,data,quote,compresion):
    old=pd.read_csv(f"{path}/BCHABC{quote}_{compresion}.csv")
    old.close_time=pd.to_datetime(old.close_time)
    old=old.set_index("close_time")[["open","low","high","close","volume"]].dropna()
    old.index += pd.Timedelta(1, unit="ms")
    return old.append(data.dropna())
    
    
def clear_data(path):
    bases=["ETC","BTC","ETH","BNB","LTC","BCH","EOS","XRP","USDT"]    
    quotes=["USDT","BTC"]
   
    for base in bases:
        for quote in quotes:
            if quote==base:
                continue
            
            
            os.system(f"rm {path}/{base}{quote}_1d.csv")
            os.system(f"rm {path}/{base}{quote}_2h.csv")
            
    os.system(f"rm {path}/USDTBTC_15t.csv")
