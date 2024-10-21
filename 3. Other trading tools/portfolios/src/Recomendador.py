import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize
from sklearn.covariance import LedoitWolf
path_to_data="./data/"


class Recomendador:
    
    def __init__(self, riesgo:float, lecturas, intervalo, currency, pendiente, factorcom):

        self.riesgo = float(riesgo)
        self.lecturas = int(lecturas)
        self.intervalo = str(intervalo)
        self.currency = currency
        self.pendiente = pendiente
        self.factorcom = factorcom

    def _get_price(self, symbol, interval):

        filename = "{}_{}.csv".format(symbol, interval)
        print(filename)
        df_price=pd.read_csv(path_to_data+filename).drop_duplicates()
        price = df_price[["open","close","low","high","Unnamed: 0"]].drop_duplicates().set_index("Unnamed: 0")
        price.index = pd.to_datetime(price.index)
        #price.index += pd.Timedelta(1, unit="ms")
        price["open"] = pd.to_numeric(price["open"])
        price = price.sort_index(ascending=True)
        price.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in price.index])
        del df_price
        return price
            
            
    def get_df(self,coins_to_consider, fecha_inicio = '', fecha_fin = ''):

        list_of_coins = coins_to_consider
        print(list_of_coins)
        #Readprices
        price={}
        for coin in list_of_coins:
            if coin=="USDT" or coin=="BTC":
                continue
            price.update({"{}USDT".format(coin):self._get_price("{}USDT".format(coin), self.intervalo)})
            if (coin.find("UP")>=0) or (coin.find("DOWN")>=0):
                continue
            price.update({"{}BTC".format(coin):self._get_price("{}BTC".format(coin), self.intervalo)})
        price.update({"BTCUSDT".format(coin):self._get_price("BTCUSDT".format(coin), self.intervalo)})
        price.update({"USDTBTC":1/price["BTCUSDT"]})
        #Build DataFrame with USDT as quote
        prices=pd.DataFrame(index=price["USDTBTC"].index)
        prices["USDT"]=1
        for coin in list_of_coins:
            if coin=="USDT":
                continue
            prices["{}".format(coin)]=price["{}USDT".format(coin)]["open"]
        #Build DataFrame with BTC as quote
        prices_BTC=pd.DataFrame(index=price["USDTBTC"].index)
        prices_BTC["BTC"]=1
        prices_BTC["USDT"]=1/price["BTCUSDT"]["open"]
    
        for coin in list_of_coins:
            if coin == "BTC":
                continue
            if coin.find('DOWN') >= 0:
                prices_BTC["{}".format(coin)]=price["{}USDT".format(coin)]["open"]/price["BTCUSDT"]["open"]
            elif coin.find('UP') >= 0:
                prices_BTC["{}".format(coin)]=price["{}USDT".format(coin)]["open"]/price["BTCUSDT"]["open"]
            else:
                prices_BTC["{}".format(coin)]=price["{}BTC".format(coin)]["open"]

        return prices, prices_BTC
    
    def generar_recomendacion(self, prices,ultima=True, start_date='', end_date='', ewm=False, total_return=False):
        """
        Devuelve un dataframe con las recomendaciones
        """
#        if prices==None:
#                
#            if self.currency == "BTC":
#                self.df = self.get_df()[1]
#            elif self.currency == "USDT":
#                self.df = self.get_df()[0]
#            else:
#                raise Exception("Currency no valida")
#            
#            self.df = self.df.dropna()
#        else:
        self.df=prices
        
        #Definir starting point
        starting_vector = tuple(np.around(np.ones(self.df.shape[1]) / self.df.shape[1], decimals=3))
        
        if start_date == '':
            fecha_inicio = self.df.index.min()
        else:
            fecha_inicio = datetime.strptime(start_date, "%Y-%m-%d") # Agregar  %H:%M:%S para cuando se pueda limitar por hora

        if end_date == '':
            fecha_fin = self.df.index.max()
        else:
            fecha_fin = datetime.strptime(end_date, "%Y-%m-%d")

        # Si las fechas no estan en el df levanto una excepcion
        if len(self.df.loc[fecha_inicio:fecha_fin]) == 0:
            raise ValueError('There is no data for those dates')

        # Comienza el calculo de final_porcentages
        final_percentages = self.df.pct_change()
        # Comienza el calculo de rolling_mean_returns

        #rolling_mean_returns = final_percentages.rolling(self.lecturas).mean()
        rolling_mean_returns = final_percentages.rolling(self.lecturas).mean()

        # limita en fechas
        rolling_mean_returns = rolling_mean_returns.loc[fecha_inicio:fecha_fin]
        #if total_return:
        all_total_returns = (self.df-(0.5)*(self.df.shift(2)+self.df.shift(1)))/self.df

        # Obtenemos las fechas validas.
        strategy_start_date = rolling_mean_returns.first_valid_index()
        strategy_end_date = rolling_mean_returns.last_valid_index()
        # calculo de covarianzas
        covariances = final_percentages.rolling(
            self.lecturas).cov().loc[strategy_start_date:strategy_end_date]

        ndim = len(rolling_mean_returns.columns)

        constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x[:ndim]) - 1.0})
        boundary = ()
        for i in range(0, ndim):
            boundary = (*boundary, (0, None))
        solutions = np.empty((0, ndim))
        # Comienza a generar la recomendacion

        rolling_mean_returns = rolling_mean_returns.dropna()
        pw = starting_vector



        for ind in rolling_mean_returns.index:
            # Crear vectores de retornos y matriz de covarianzas
            returns = np.array(rolling_mean_returns.loc[ind])*(365*12)
            covariance = np.matrix(covariances.loc[ind])*(365*12)
#            ###
#            X = np.random.multivariate_normal(mean=returns,cov=covariance,size=50)
#            cov = LedoitWolf().fit(X)
#            covariance_lw = cov.covariance_
#            ###

            total_returns = np.array(all_total_returns.loc[ind])
            # Definir la función a optimizar (como estamos minimizando, hay que usar -f(x))

            def portfolio_function(x):
                comision = (np.where((x-pw) < 0, 0, (x-pw))).sum()
                #comision=0
                return float(- np.dot(returns, x)
                             + self.riesgo * np.dot(x, np.dot(covariance, x).T)
                             + self.pendiente*np.dot(total_returns, x)
                             + self.factorcom*comision)

            problem = optimize.minimize(portfolio_function,
                                        starting_vector,
                                        bounds=boundary,
                                        constraints=constraint, method="SLSQP"
                                        #options={'maxiter': 10000, 'ftol': 1e-05, 'iprint': 1,
                                         #        'disp': False, 'eps': 0.500000e-7}#1.4901161193847656e-08}
                                       )
            # Apilemos los vectores de soluciones por cada día
            solutions = np.append(solutions, np.array(
                problem.x.reshape(1, ndim)), axis=0)
            pw = np.array(problem.x.reshape(1, ndim))[0, :]
            
        # Transformamos el vector de soluciones a un DataFrame
        weights = pd.DataFrame(solutions,
                               index=pd.to_datetime(
                                   rolling_mean_returns.index, unit='ms'),
                               columns=list(self.df.columns))
        #weights.iloc[self.df_columns]
        weights = weights.apply(lambda x: x/weights.sum(axis=1))
        weights = np.round(weights,3)
        return weights
