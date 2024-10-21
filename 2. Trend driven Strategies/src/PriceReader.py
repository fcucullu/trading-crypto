#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:59:13 2020

@author: farduh
"""
import pandas as pd




class PriceReader:
    
    def __init__(self,path_to_data="data/"):
        self.path_to_data=path_to_data
    
    
    def get_price(self, symbol, interval,index_column="close_time"):

        filename = "{}_{}.csv".format(symbol, interval)
        print(filename)
        df_price=pd.read_csv(self.path_to_data+filename).drop_duplicates()
        price = df_price[["open","close","low","high","volume",index_column]].drop_duplicates().set_index(index_column)
        price.index = pd.to_datetime(price.index)
        price.index += pd.Timedelta(1, unit="ms")
        price["close"] = pd.to_numeric(price["close"])
        price = price.sort_index(ascending=True)
        price.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in price.index])
        del df_price
        return price
            
            
    def get_price_df(self,intervalo,col="close", fecha_inicio = '', fecha_fin = ''):

        price={"BNBBTC":self.get_price("BNBBTC", intervalo),
            "ETHBTC":self.get_price("ETHBTC", intervalo),
            "LTCBTC":self.get_price("LTCBTC", intervalo),
          #  "XRPBTC":self.get_price("XRPBTC", intervalo),
          #  "TRXBTC":self.get_price("TRXBTC", intervalo),
            "BNBUSDT":self.get_price("BNBUSDT", intervalo),
            "ETHUSDT":self.get_price("ETHUSDT", intervalo),
            "LTCUSDT":self.get_price("LTCUSDT", intervalo),
          #  "XRPUSDT":self.get_price("XRPUSDT", intervalo),
          #  "TRXUSDT":self.get_price("TRXUSDT", intervalo),
            "BTCUSDT":self.get_price("BTCUSDT", intervalo),
            }
        price.update({"USDTBTC":1/price["BTCUSDT"]})

        price["USDTBTC"]["return"]=price["USDTBTC"]["close"].pct_change()
        
        prices=pd.DataFrame(index=price["USDTBTC"].index)
        
        prices["USDT"]=1
        prices["BTC"]=price["BTCUSDT"][col]
        prices["ETH"]=price["ETHUSDT"][col]
        prices["BNB"]=price["BNBUSDT"][col]
        prices["LTC"]=price["LTCUSDT"][col]

        prices_BTC=pd.DataFrame(index=price["USDTBTC"].index)
        prices_BTC["BTC"]=1
        prices_BTC["USDT"]=1/price["BTCUSDT"][col]
        prices_BTC["ETH"]=price["ETHUSDT"][col]
        prices_BTC["BNB"]=price["BNBUSDT"][col]
        prices_BTC["LTC"]=price["LTCUSDT"][col]
        
        return prices, prices_BTC
    
