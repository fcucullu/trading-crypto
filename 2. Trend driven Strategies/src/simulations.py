#By Francu

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pylab 
import statsmodels.api as sm
from itertools import groupby
from operator import itemgetter
import statsmodels.graphics.tsaplots as tsa
from scipy.optimize import curve_fit
from scipy.stats import t, norm, f_oneway, shapiro, probplot, levene, kruskal
from sklearn.tree import DecisionTreeRegressor

############################################################################
############################################################################

class GeometricBrownianMotionSimulator():
    
    def __init__(self, S0, dt, mu, sigma):
        
        self.S0 = S0
        self.dt = dt
        self.mu = mu
        self.sigma = sigma

    def simulate(self, length):
        
        '''
        Esta función genera una simulacion de un proceso browniano geométrico estándard.
        Supone que los retornos se comportan normalmente.
        
        Inputs:
            length: es la cantidad de observaciones (filas) que se crearán.
            S0: es el precio de inicio. De donde partirán las simulaciones (primer fila en todas las columnas).
            dt: es el avance temporal en términos anuales de cada observación. dt = 1/obs_x_año.
            mu: es la media estimada de la serie.
            sigma: es la volatilidad estimada de la serie.
    
        Output:
            serie_price: son los precios simulados.
        
        '''
        
        normal = np.random.normal(size=(1, length-1))[0] #-1 por que el S0 ocupa un espacio
        serie_price = []
        serie_price.append(self.S0)
        for i in range(length-1): 
            new_price = serie_price[i]*np.exp((self.mu-0.5*self.sigma*self.sigma) * self.dt + self.sigma * np.sqrt(self.dt) * normal[i]) 
            serie_price.append(new_price)                                                       #mov Browniano           
        return serie_price


    def simulate_from_last(self, last, length):
        
        '''Genera varias simulaciones para el tiempo t+1 partiendo del last'''
        normal = np.random.normal(size=(1, length-1))[0] #-1 por que el S0 ocupa un espacio
        serie_price = []
        serie_price.append(last)
        for i in range(length-1): 
            new_price = serie_price[i]*np.exp((self.mu-0.5*self.sigma*self.sigma) * self.dt + self.sigma * np.sqrt(self.dt) * normal[i]) 
            serie_price.append(new_price)                                                       #mov Browniano           
        return serie_price
    
    
    def simulate_n_times(self, number_simulations, length):
        '''
        Esta función genera N simulaciones de un proceso browniano geométrico estándard.
        Supone que los retornos se comportan normalmente.
        
        Inputs:
            simulaciones: es la cantidad de simulaciones (columnas) que crearán.
            length: es la cantidad de observaciones (filas) que se crearán.
            S0: es el precio de inicio. De donde partirán las simulaciones (primer fila en todas las columnas).
            dt: es el avance temporal en términos anuales de cada observación. dt = 1/obs_x_año.
            mu: es la media estimada de la serie.
            sigma: es la volatilidad estimada de la serie.
            graph: False. Si es True devolverá un grafico en niveles y otro en diferencias.
    
        Output:
            monte_carlo: es la tabla de los precios simulados.
    
        '''
        monte_carlo = pd.DataFrame()
        for k in range(number_simulations): 
            monte_carlo[k] = self.simulate(length)
        return monte_carlo



        
class MertonJumpDiffusionSimulator():
    
    def __init__(self, S0, dt, muhat, sigmahat, Lambdahat, mu_jhat, sigma_jhat):
        self.S0 = S0
        self.dt = dt
        self.muhat = muhat
        self.sigmahat = sigmahat
        self.Lambdahat = Lambdahat
        self.mu_jhat = mu_jhat
        self.sigma_jhat = sigma_jhat
    
    def simulate(self, length):
        '''
        Esta función genera una simulacion de un proceso de Jump-Diffusion.
        Supone que los retornos siguen dos procesos estocásticos combinados. 
        Un primer proceso es el Browniano geométrico estándar, es decir que supone
        que los retornos tienen una parte central que se comporta normalmente.
        El segundo proceso es un proceso de Poisson que supone que arriba del anterior,
        tambien hay eventos extraordinarios que provocan outliers y engordan las colas.
        
        Inputs:
            length: es la cantidad de observaciones (filas) que se crearán.
            S0: es el precio de inicio. De donde partirán las simulaciones (primer fila en todas las columnas).
            dt: es el avance temporal en términos anuales de cada observación. dt = 1/obs_x_año.
            muhat: es la media estimada de la serie.
            sigmahat: es la volatilidad estimada de la serie.
            Lambdahat: es la probabilidad de ocurrencia de un evento extremo en un año.
            mu_jhat: es la media estimada de los retornos del proceso Poisson de los eventos extremos
            sigma_jhat: es la volatilidad estimada de los retornos del proceso Poisson de los eventos extremos
    
        Output:
            serie_price: son los precios simulados.
        '''
        serie_price = []
        serie_price.append(self.S0)
        normal_central = np.random.normal(size=(1, length-1))[0]
        normal_jump = np.random.normal(size=(1, length-1))[0]
        poisson = np.random.poisson(self.Lambdahat * self.dt, size=(1, length-1))[0]
        for i in range(length-1): 
            jump = self.mu_jhat * poisson[i] + self.sigma_jhat * np.sqrt(poisson[i]) * normal_jump[i]
            new_price = serie_price[i]*np.exp((self.muhat-0.5*self.sigmahat*self.sigmahat) * self.dt + self.sigmahat * np.sqrt(self.dt) * normal_central[i] + jump)
            serie_price.append(new_price)
        return serie_price
    
    def simulate_from_last(self, last, length):
        
        '''Genera varias simulaciones para el tiempo t+1 partiendo del last'''
        
        serie_price = []
        serie_price.append(last)
        normal_central = np.random.normal(size=(1, length-1))[0]
        normal_jump = np.random.normal(size=(1, length-1))[0]
        poisson = np.random.poisson(self.Lambdahat * self.dt, size=(1, length-1))[0]
        for i in range(length-1): 
            jump = self.mu_jhat * poisson[i] + self.sigma_jhat * np.sqrt(poisson[i]) * normal_jump[i]
            new_price = serie_price[i]*np.exp((self.muhat-0.5*self.sigmahat*self.sigmahat) * self.dt + self.sigmahat * np.sqrt(self.dt) * normal_central[i] + jump)
            serie_price.append(new_price)
        return serie_price
        
        
    def simulate_n_times(self, number_simulations, length):
        '''
        Esta función genera simulaciones de un proceso de Jump-Diffusion.
        Supone que los retornos siguen dos procesos estocásticos combinados. 
        Un primer proceso es el Browniano geométrico estándar, es decir que supone
        que los retornos tienen una parte central que se comporta normalmente.
        El segundo proceso es un proceso de Poisson que supone que arriba del anterior,
        tambien hay eventos extraordinarios que provocan outliers y engordan las colas.
        
        Inputs:
            simulaciones: es la cantidad de simulaciones (columnas) que crearán.
            length: es la cantidad de observaciones (filas) que se crearán.
            S0: es el precio de inicio. De donde partirán las simulaciones (primer fila en todas las columnas).
            dt: es el avance temporal en términos anuales de cada observación. dt = 1/obs_x_año.
            muhat: es la media estimada de la serie.
            sigmahat: es la volatilidad estimada de la serie.
            Lambdahat: es la probabilidad de ocurrencia de un evento extremo en un año.
            mu_jhat: es la media estimada de los retornos del proceso Poisson de los eventos extremos
            sigma_jhat: es la volatilidad estimada de los retornos del proceso Poisson de los eventos extremos
    
        Output:
            monte_carlo_jump: es la tabla de los precios simulados.
        '''
        
        #Simulacion + serie real
        monte_carlo_jump = pd.DataFrame()
        for k in range(number_simulations):
            monte_carlo_jump[k] = self.simulate(length)
        return monte_carlo_jump
    

    

class Graphs():
    '''En esta clase estan todas las funciones que grafican la información'''
    
    def graph_price_time(self, serie_price, label:str):
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle(label, fontsize=22, fontweight="bold")
        plt.plot(serie_price, label='Simulation', lw=3, alpha=0.8)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.show()
        
    def graph_return_time(self, serie_price, label:str):
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle(label, fontsize=22, fontweight="bold")
        plt.plot(returns, label='Simulation', lw=3, alpha=0.8)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Returns', fontsize=15)
        plt.show()
    
    def graph_return_time_with_3std(self, serie_price, label:str):
        '''Identificacion de outliers segun Cartea y Figueroa (2005)'''
        tabla = pd.DataFrame()
        tabla['returns'] = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        tabla['mean_return'] = tabla['returns'].mean()        
        tabla['3sd'] = 3*tabla['returns'].std()
        tabla['-3sd'] = -3*tabla['returns'].std()
        plt.figure(figsize=(12, 7))
        arr0 = plt.plot(tabla['returns'], lw=3, alpha=0.8)
        arr1 = plt.plot(tabla['mean_return'], lw=3, alpha=0.8)
        arr2 = plt.plot(tabla['3sd'], lw=3, alpha=0.8)
        arr3 = plt.plot(tabla['-3sd'], lw=3, alpha=0.8)
        plt.legend([arr0,arr1,arr2,arr3],['Retornos','Media','+3std','-3std'], loc=4, fontsize=12)
        plt.ylabel('Retornos logarítmicos ', fontsize=15)
        plt.xlabel('Tiempo', fontsize=15)
        plt.title(label, fontsize=22, fontweight="bold")
        plt.show()
        
    def graph_qqplot(self, serie_price, label:str):
        '''QQ Plot de la libreria scipy'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        plt.figure(figsize=(12, 7))    
        probplot(returns, dist='norm', plot=pylab)
        plt.ylabel('Observaciones ordenadas', fontsize=15)
        plt.xlabel('Cuantiles teóricos', fontsize=15)
        plt.title(label, fontsize=22, fontweight="bold")
        pylab.show()

    def graph_comparison_pdf_returns_normal(self, serie_price):
        '''Grafica dos funciones de probabilidad (PDF)'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle('Comparación de densidades',fontsize=22, fontweight="bold")
        sns.distplot(returns, hist=False, kde=True, color = 'red', 
                       label='Retornos observados', kde_kws={'linewidth': 4})
        normal = np.random.normal(returns.mean(), returns.std(), len(returns))
        sns.distplot(normal, hist=False, kde=True, color = 'green', 
                     label='Normal', kde_kws={'linewidth': 4})
        plt.xlabel('Dominio de los retornos',fontsize=15)
        plt.ylabel('Densidad de probabilidad',fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.show()
    
    def graph_comparison_pdf_simulations(self, serie_price, simulation_price):
        '''Grafica dos funciones de probabilidad (PDF)'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        simulated_returns = PriceWorker.calculate_logaritmic_returns(self, pd.DataFrame(simulation_price))
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle('Comparación de densidades',fontsize=22, fontweight="bold")
        sns.distplot(returns, hist=False, kde=True, color = 'red', 
                       label='Retornos observados', kde_kws={'linewidth': 4})
        sns.distplot(simulated_returns, hist=False, kde=True, color = 'green', 
                     label='Simulacion', kde_kws={'linewidth': 4})
        plt.xlabel('Dominio de los retornos',fontsize=15)
        plt.ylabel('Densidad de probabilidad',fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.show()    
    
    def graph_autocorrelation(self, serie_price):
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        # autocorrelation
        fig = sm.graphics.tsa.plot_acf(returns, lags=40)
        plt.show()
        # partial autocorrelation
        fig = sm.graphics.tsa.plot_pacf(returns, lags=40)
        del fig
        plt.show()
        
    def graph_candlesticks(self, ohlc_data, name_diff=''):
        
        import plotly.graph_objects as go
        import plotly.offline as py_offline
        
        data = [ 
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data['open'+name_diff],
                high=ohlc_data['high'+name_diff],
                low=ohlc_data['low'+name_diff],
                close=ohlc_data['close'+name_diff]
                            )
                ]

        fig = go.Figure(data=data)
        
        py_offline.plot(fig, filename='Candle Stick')


    def check_simulation_correlation(self,data,list_simulations,savefig=False):
        simulations = list_simulations.copy()
        hdata = data.copy()
        for sim in simulations:
            sim["returns"]=sim["close"].pct_change()
            sim["upper"] = ((sim["high"] / np.maximum(sim["open"],sim["close"])) /
                (np.maximum(sim["open"],sim["close"]) / np.minimum(sim["open"],sim["close"]))) -1
            sim["lower"] = ((np.minimum(sim["open"],sim["close"]) / sim["low"]) /
             (np.maximum(sim["open"],sim["close"]) / np.minimum(sim["open"],sim["close"]))) -1
            sim["log_volume"] = np.log(sim["volume"])
        hdata["returns"]=hdata["close"].pct_change()
        hdata["upper"] = ((hdata["high"] / np.maximum(hdata["open"],hdata["close"])) /
            (np.maximum(hdata["open"],hdata["close"]) / np.minimum(hdata["open"],hdata["close"]))) -1
        hdata["lower"] = ((np.minimum(hdata["open"],hdata["close"]) / hdata["low"]) /
         (np.maximum(hdata["open"],hdata["close"]) / np.minimum(hdata["open"],hdata["close"]))) -1
        hdata["log_volume"] = np.log(hdata["volume"])
        l=1
        
        plt.figure(figsize=(20,12))
        for var in ["returns","upper","lower","log_volume"]:
            print(var)
            corr=[]
            for sim in simulations:
                corr.append(np.corrcoef(sim[var].shift(1)[2:],sim[var][2:])[1,0])
            corr_data=np.corrcoef(hdata[var].shift(1)[2:],hdata[var][2:])[1,0]
            print(np.mean(corr),corr_data)
            plt.subplot(2,2,l)   
            for sim in simulations:
                sim[var].hist(bins=50,density=1,histtype="step")
            hdata[var].hist(bins=50,density=1,histtype="step",color="black")
            plt.title(f"{var}\n corr t vs t-1, sim: {round(np.mean(corr),4)} data: {round(corr_data,4)}")
            
            l+=1
            
        if savefig:
            plt.savefig("check_simulation_correlations.pdf")
        plt.show()

class PriceExplorer():
    '''En esta clase estan todas las funciones que analizan la información'''
   
    def check_differences_in_samples_MEAN_non_normal(self, distribution1, distribution2):
        ''' test >= p_value : misma media
            test < p_value : distinta media
        '''
        return kruskal(distribution1, distribution2, nan_policy='omit')[1]
            
    def check_differences_in_samples_MEAN_normal(self, distribution1, distribution2):
        ''' test >= p_value : misma media
            test < p_value : distinta media
        '''
        return f_oneway(distribution1, distribution2, nan_policy='omit')[1]

    def check_differences_in_samples_VARIANCE(self, distribution1, distribution2):
        ''' test >= p_value : misma volatilidad
            test < p_value : distinta volatilidad
        '''
        return levene(distribution1, distribution2, center='mean')[1]
        
    def compare_distributions_non_normal(self, dist_origin, dist_addition, p_value):
        ''' 
        Esta funcion compara dos distribuciones para saber si se pueden
        analizar en conjunto y generar una simulacion considerando ambas. 
        Para ello primero testea la homocedasticidad y luego testea la semejanza de su media.
        
        return:
            True si se pueden analizar en conjunto.
            False si no se pueden analizar en conjunto.
        '''
        filter_var = self.check_differences_in_samples_VARIANCE(dist_origin, dist_addition)
        return filter_var >= p_value
    
    def get_sample_for_simulations_non_normal(self, serie_price, periods, p_value):
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        time = int( periods / 365*7 ) 
        end = len(returns) #last obs
        start = end - time #beginning of last week 
        sample = returns[start:end] #initial sample
        mult = 1
        
        while True == True:
            sample_add = returns[start-time*mult : end-time*mult]
            if self.compare_distributions_non_normal(sample, sample_add, p_value) == False or len(sample_add) == 0:
                break
            sample = returns[start-time*mult:end]
            mult += 1
        return serie_price[start-time*(mult-1) : end+1]
                 
    def check_stationarity(self, serie_price, significance=0.05):
        '''Test Dickey-Fuller Aumentado para checkear estacionariedad'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        test_value, pvalue = adfuller(returns)[0:2]
        #H0: raiz unitaria, no estacionario; Ha: Ausencia de raiz unitaria, estacionario.
        if significance > pvalue:
            print('\n *** Los retornos son estacionarios ***')
        else:
            print('\n *** Los retornos NO son estacionarios ***')
        
    def check_normality(self, serie_price, significance=0.05):
        ''' Test de normalidad de Shapiro '''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        test_value, pvalue = shapiro(returns)
        #H0: Normalidad; Ha: Ausencia de normalidad.
        if significance < pvalue:
            print('\n *** Los retornos originales se comportan normalmente ***')
        else:
            print('\n *** Los retornos originales NO se comportan normalmente ***')
    
    def full_timeseries_analysis(self, serie_price):      
        '''
        Esta función recorre todos aquellos factores que deben controlarse
        cuando se trabaja con series temporales.
        
        Inputs:
            serie_prices: es la pd.DataFrame con la información de la serie temporal.
            
        Output:
            Gráficos
        
        '''
        print('\n ########################################################')
        print('\n ***   A N Á L I S I S      E X P L O R A T O R I O  V***')
        print('\n ########################################################')
        Graphs.graph_price_time(self, serie_price, 'Evolución de precios')
        Graphs.graph_return_time_with_3std(self, serie_price, 'Identificación de outliers')
        PriceExplorer.check_stationarity(self, serie_price)
        Graphs.graph_qqplot(self, serie_price, 'QQ Plot de Retornos vs Normal')
        Graphs.graph_comparison_pdf_returns_normal(self, serie_price)
        PriceExplorer.check_normality(self, serie_price)
        Graphs.graph_autocorrelation(self, serie_price)
        
    
class PriceWorker():  
    '''En esta clase estan todas las funciones que modifican la información'''
    
    def revert_returns_to_prices(self, serie_returns, S0=100):
        return (serie_returns + 1).cumprod() * S0
    
    def calculate_logaritmic_returns(self, serie_price):
        returns = np.log (serie_price.shift(-1) / serie_price )
        returns = returns [:-1]
        return returns
    
    def calculate_returns(self, serie_price):    
        returns = serie_price.shift(-1) / serie_price - 1
        returns = returns [:-1]
        return returns    
    
    def calculate_mean_and_std(self, serie_price):
        returns = self.calculate_logaritmic_returns(serie_price)
        return returns.mean(), returns.std()
    
    def filter_carterafigueroa(self, serie_price, std_bias=3):
        ''' 
        Esta funcion analiza los valores extremos segun el proceso que
        proponen los autores.
        El proceso consiste en filtrar las variaciones que sobrepasen los 
        +/- 3std y almacenarlas en una base aparte. Luego se vuelven a calcular
        los desvíos y se repite el filtrado sucesivamente hasta que no existan
        mas variaciones a filtrar.
        Al final de este proceso se tendrá una tabla con las observaciones centrales
        (se supone que son normales, pero hay que verificarlo) y otra tabla
        diferente con las variaciones extremas. 
        Se podrá calcular cuál es el % de probabilidad que ocurran eventos extremos
        y se deberá analizar qué distribucion siguen estas observaciones.
        
        Inputs:
            serie_price: información a filtrar
            
        Output:
            central_process: tabla sin variaciones extremas, proceso central
            outliers: variaciones extremas filtradas
            '''      
        print('\n-----------------------------------------')
        print('Analizo JUMPS segun Cartea-Figueroa (2005)')
        print('-----------------------------------------\n')
     
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        central_process_returns = pd.DataFrame(returns) 
        central_process_returns['Filter'] = ''  
    
        filters_done = 1
        outliers_returns = pd.DataFrame()
        BREAK = False
        
        while BREAK == False:
            deviation = central_process_returns.std()*std_bias
            for i in range(len(central_process_returns)): 
                central_process_returns['Filter'][i] = 'Filter' if abs(central_process_returns[serie_price.name][i]) >= deviation[0] else 'ok'
            Filter =  central_process_returns['Filter']=='ok'
            data_filtered = central_process_returns[central_process_returns['Filter']=='Filter'] 
            print('El filtro ({}) se eliminó un {}% de los datos originales'.format(filters_done, round(len(data_filtered)/len(returns)*100,2)))
        
            central_process_returns = central_process_returns[Filter]
            outliers_returns = pd.concat([outliers_returns, data_filtered]).sort_index()
            filters_done += 1
        
            if len(data_filtered) == 0:
                BREAK = True   
        
        del central_process_returns['Filter']
        del outliers_returns['Filter']
        
        central_process_price = self.revert_returns_to_prices(central_process_returns.iloc[:,0], serie_price[0])
        outliers_price = self.revert_returns_to_prices(outliers_returns.iloc[:,0], serie_price[0])
        
        return central_process_price, outliers_price #Para que salgan pd.Series
    
    def calculate_parameters_mertonjumpdiffusion_aproximation(self, central_process_prices, outliers_prices, periods):
        '''
        Esta función calcula por una aproximación matemática los parámetros para
        reproducir un proceso de Merton Jump-Diffusión.
        
        Inputs:
            central_process_returns: información del proceso Normal central. Es la tabla con los datos no-extremos
            outliers_returns: información del proceso de Poisson de los outliers_returns. Es la tabla con los datos extremos.
            periods: son la cantidad de observaciones en un año. Sirve para calcular métricas anuales.
            
        Outputs:
            muhat: es la media estimada de los retornos del proceso Normal central 
            sigmahat: es la volatilidad estimada de los retornos del proceso Normal central 
            Lambdahat: es la probabilidad de ocurrencia de un evento extremo en un año.
            mu_jhat: es la media estimada de los retornos del proceso Poisson de los eventos extremos
            sigma_jhat: es la volatilidad estimada de los retornos del proceso Poisson de los eventos extremos
            
        '''     
        dt = 1/periods
        central_process_returns = PriceWorker.calculate_logaritmic_returns(self, central_process_prices)
        outliers_returns = PriceWorker.calculate_logaritmic_returns(self, outliers_prices)
        
        #Parametros de la difusion
        muhat = (2 * central_process_returns.mean() + central_process_returns.var() * dt) / 2 * dt
        sigmahat = np.sqrt( central_process_returns.var() / dt )
        
        #Parametros del jump
        Lambdahat = int(len(outliers_returns) / (len(central_process_returns)+len(outliers_returns)) * periods) #Cantidad de jumps por año
        
        mu_jhat = outliers_returns.mean() - (muhat - 0.5 * sigmahat * sigmahat) * dt
        sigma_jhat = np.sqrt( outliers_returns.var() - sigmahat * sigmahat * dt )  
    
        return muhat, sigmahat, Lambdahat, mu_jhat, sigma_jhat
    
    def rebuild_tabla_by_index(self, table_origin, table_filtered):
        return table_origin[table_origin.index.isin(table_filtered.index)]

    def wicks_temporal_fix(monte_carlo):
        close_great_equal_open = monte_carlo['close'] >= monte_carlo['open'] 
        high_lower_close = monte_carlo['high'] < monte_carlo['close'] 
        high_lower_open = monte_carlo['high'] < monte_carlo['open'] 
        low_higher_close = monte_carlo['low'] > monte_carlo['close'] 
        low_higher_open = monte_carlo['low'] > monte_carlo['open'] 
        monte_carlo.loc[close_great_equal_open & high_lower_close, 'high'] = monte_carlo['close']
        monte_carlo.loc[close_great_equal_open & low_higher_open, 'low'] = monte_carlo['open']
        monte_carlo.loc[~close_great_equal_open & high_lower_open, 'high'] = monte_carlo['open']
        monte_carlo.loc[~close_great_equal_open & low_higher_close, 'low'] = monte_carlo['close']
        return monte_carlo


class VolumeSimulator:
    def __init__(self,
                 func_to_fit,
                 parameter_names,
                 func_parameters,
                 path_to_plots="."):
        """
        This class simulates volume based on high-low price differences
        Input:
            - path_to_plots: string with the directory path on which the figures are saved.
            - func_to_fit: function to adjust the volume distribution.
            - parameters_names: list of parameters names in the same order that the fit function return them.
            - func_parameters: dictionary containing functions to adjust parameters as a function of diffHL.
            
        """
        self.path_to_plots=path_to_plots
        self.func_to_fit=func_to_fit
        self.parameter_names=parameter_names
        self.func_parameters=func_parameters
        
    def save_histograms(self,nbins,y_,optimal_parameters,tag):
        """
        Save histograms and function distribution figures
        """
        numb_subplots=int(np.sqrt(nbins))+1
        
        plt.figure(figsize=(numb_subplots*2.5,numb_subplots*2.5))        
        for isubplot in range(nbins):
            plt.subplot(numb_subplots,numb_subplots,isubplot+1)
            y_[isubplot].hist(bins=40,density=1,range=(y_[isubplot].min(),y_[isubplot].max()))
            plt.plot(np.arange(y_[isubplot].min(),y_[isubplot].max(),0.01),
                     self.func_to_fit.pdf(np.arange(y_[isubplot].min(),
                                                    y_[isubplot].max(),
                                                    0.01),
                                            **optimal_parameters.iloc[isubplot].to_dict()))
            plt.xlabel("volume")
        
        plt.savefig(f"{self.path_to_plots}/histograms_{tag}.pdf")
    
    def save_plot(self,optimal,final_parameters,parameters_to_adjust):
        """
        Save parameter fit plots
        """
        for name in parameters_to_adjust:
            plt.figure(figsize=(10,8))
            optimal[name].plot(marker=".",linestyle="")
            plt.plot(np.arange(optimal.index.min(),optimal.index.max(),0.1),
                     self.func_parameters[name](np.arange(optimal.index.min(),optimal.index.max(),0.1)
                     ,*final_parameters[name]))
            plt.xlabel("high-low differences")
            plt.xlabel(f"{name} value")
            plt.savefig(f"{self.path_to_plots}/ajuste_{name}.pdf")
    
    def fit_distributions(self,
                          log_volume,
                          log_hl,
                          x_hl,
                          nbins,
                          parameters_fixed={},
                          savefig=False,
                          tag=""):
        """
        Function to fit volumen distributions in several bins
        """
        range_nbins=np.arange(nbins)
        y_vec=[log_volume[(x_hl[i]<log_hl) & (log_hl<x_hl[i+1])] for i in range_nbins]
        x_vec=[log_hl[(x_hl[i]<log_hl) & (log_hl<x_hl[i+1])].mean() for i in range_nbins]
        optimal=[dict(zip(self.parameter_names,
                          self.func_to_fit.fit(y_vec[i],**parameters_fixed))
                    )for i in range_nbins]
        optimal=pd.DataFrame(optimal,index=x_vec)
        
        if savefig==True:
            self.save_histograms(nbins,y_vec,optimal,tag)
        
        return optimal
            
    def get_parameters_volume_simulation(self,prices,nbins=10,savefig=False):

        """
        This function returns the parameters using to model volume distribution as a function of the diference between high and low prices (candlestick_range)
        
        Inputs:
            - prices: DataFrame with high, low prices and volume.
            - nbins: number of candlestick range bins. It must be greater than the parameters in the parameter functions.
            - savefig: True if you want to save figures in the optimization.
            
        Output:
            - parameters to simulate the volume from candlestick ranges.
        
        """
        candlestick_range = 10000*(prices["high"]-prices["low"]).dropna()
        volume = prices["volume"].dropna()
        parameters_to_adjust = list(self.func_parameters.keys())
        log_candlestick_range = np.log(1+candlestick_range)
        log_volume = np.log(volume)
        x_candlestick_range = np.quantile(log_candlestick_range.values,np.linspace(0,1,nbins+1))

        #primera optimización sin fijar parámetros
        optimal=self.fit_distributions(
                          log_volume,
                          log_candlestick_range,
                          x_candlestick_range,
                          nbins,
                          parameters_fixed={},
                          savefig=savefig,
                          tag="all_parameters_free")
        if self.func_parameters:
            #segunda optimización fijando parámetros
            parameters_fixed={"f"+name:optimal[name].median() for name in self.parameter_names if name not in parameters_to_adjust}
            optimal=self.fit_distributions(
                              log_volume,
                              log_candlestick_range,
                              x_candlestick_range,
                              nbins,
                              parameters_fixed=parameters_fixed,
                              savefig=savefig,
                              tag="some_parameters_fixed")        
            #aproximando parámetros por funciones
            final_parameters={name:optimal[name].median()             
                                    if not name in parameters_to_adjust
                                    else curve_fit(self.func_parameters[name],
                                                   optimal.index,
                                                   optimal[name].values)[0]
                                    for name in self.parameter_names
                                    }
            if savefig==True:
                self.save_plot(optimal,final_parameters,parameters_to_adjust)
        else:
            final_parameters={
                    name:optimal[name].median() for name in self.parameter_names}
            
        return final_parameters
    
    def simulated_volumen_from_candlestick_range(self,candlestick_range,parameters):
        """
        Method to simulate the volume from the differences between high and low prices (candlestick range)
        Input:
            - candlestick_range: Series with simulated candlestick ranges.
            - parameters: parameters of the volume distribution.
        
        """
        log_candlestick_range=np.log(1+10000*candlestick_range)
        dict_to_rvs={name:
            parameters[name] if name not in list(self.func_parameters.keys())
            else self.func_parameters[name](log_candlestick_range,*parameters[name]) 
            for name in self.parameter_names}
        log_random_volume=self.func_to_fit.rvs(**dict_to_rvs)
        
        return (np.exp(log_random_volume))
      
    
class WicksSimulator():
    def __init__(self, data):
        self.data = data
        self.identify_wicks()
        self.fit_wicks()        
        
    def identify_wicks(self):
        upper_green, lower_green, upper_red, lower_red = [], [], [], []
                
        for i in range(len(self.data)):
            _close = self.data.iloc[i].close
            _open = self.data.iloc[i].open  
            _high = self.data.iloc[i].high
            _low = self.data.iloc[i].low     
                
            if _close >= _open:
                upper_green.append( ((_high / _close) / (_close / _open)) -1)        
                lower_green.append( ((_open / _low) / (_close / _open)) -1)        
            else:
                upper_red.append( ((_high / _open) / (_open / _close)) -1)        
                lower_red.append( ((_close / _low) / (_open / _close)) -1)        

        self.empirical_wicks = {
            'upper_green': pd.DataFrame(upper_green),
            'lower_green': pd.DataFrame(lower_green),
            'upper_red': pd.DataFrame(upper_red),
            'lower_red': pd.DataFrame(lower_red)}
                         
    def fit_wicks(self):     
        self.fitted_wicks = {}     
        for k,v in self.empirical_wicks.items():
            df, loc, scale = t.fit(self.empirical_wicks[k][0])          
            self.fitted_wicks[k] = pd.DataFrame({'df':[df], 'loc':[loc], 'scale':[scale]})         
    
    def simulate_wicks(self, length):
        simulated_wicks = {}       
        for k,v in self.fitted_wicks.items():           
            simulated_wicks[k] = t.rvs(loc=  self.fitted_wicks[k]['loc'][0],
                                       scale=self.fitted_wicks[k]['scale'][0],
                                       df=   self.fitted_wicks[k]['df'][0],
                                       size=length)      
        return pd.DataFrame(simulated_wicks)
    
    
class SimulateMertonOHLC(MertonJumpDiffusionSimulator, WicksSimulator, VolumeSimulator, PriceWorker):
    def __init__(self, data, periods):
        self.data = data
        self.periods = periods
        WicksSimulator.__init__(self, self.data)
        self.central_process, self.outliers = PriceWorker.filter_carterafigueroa(self, self.data.close)
        self.muhat, self.sigmahat, self.Lambdahat, self.mu_jhat, self.sigma_jhat = PriceWorker.calculate_parameters_mertonjumpdiffusion_aproximation(self, self.central_process, self.outliers, self.periods)
        MertonJumpDiffusionSimulator.__init__(self, self.data.close[0], 1/self.periods, self.muhat, self.sigmahat, self.Lambdahat, self.mu_jhat, self.sigma_jhat)
        self.volume_simulator = VolumeSimulator(norm,["loc","scale"],{"loc":self.lineal})
        lenght_data = len(self.data)
        bins = 2
        data_lenght_in_one_bin = 10000
        while data_lenght_in_one_bin < 150:
            bins += 1
            data_lenght_in_one_bin = lenght_data/(bins+1)
        self.volumen_simulator_parameters = self.volume_simulator.get_parameters_volume_simulation(self.data,nbins=bins)
        
    def lineal(self,x,c1,x0):
        return c1*(x-x0)
        
    def simulate_ohlc_n_times(self, number_simulations, length, volume=False):      
        '''Crea una lista de tablas OHLC'''
        list_simulations = []       
        for k in range(number_simulations): 
            monte_carlo = pd.DataFrame()           
            close_list = self.simulate(length)
            open_list = [self.data.close[0], self.data.close[0]]
            open_list = open_list + close_list[1:-1]
            monte_carlo['open'] = open_list
            monte_carlo['close'] = close_list           
            wicks = self.simulate_wicks(length)
            close_great_equal_open = monte_carlo['close'] >= monte_carlo['open'] 
            monte_carlo.loc[close_great_equal_open, 'high'] =  ( (1+wicks['upper_green']) * monte_carlo['close'] * monte_carlo['close'] ) / monte_carlo['open']
            monte_carlo.loc[close_great_equal_open, 'low'] = ( monte_carlo['open'] * monte_carlo['open'] ) / ( (1+wicks['lower_green']) * monte_carlo['close'] )
            monte_carlo.loc[~close_great_equal_open, 'high'] = ( (1+wicks['upper_red']) * monte_carlo['open'] * monte_carlo['open'] ) / monte_carlo['close']
            monte_carlo.loc[~close_great_equal_open, 'low'] = ( monte_carlo['close'] * monte_carlo['close'] ) / ( (1+wicks['lower_red']) * monte_carlo['open'] )
            #Control mechas
            monte_carlo = PriceWorker.wicks_temporal_fix(monte_carlo)
            if volume:
                monte_carlo['volume'] = self.volume_simulator.simulated_volumen_from_candlestick_range(monte_carlo['high']-monte_carlo['low'],self.volumen_simulator_parameters)
            list_simulations.append(monte_carlo)                                
        return list_simulations
       
    def simulate_ohlc_n_times_with_variable_mean(self, mean_array, number_simulations, length, volume=False):
        if len(mean_array) != number_simulations:
            print('No hay misma cantidad de medias y simulaciones')
            return
        '''Crea una lista de tablas OHLC'''
        list_simulations = []       
        for k in range(number_simulations): 
            monte_carlo = pd.DataFrame() 
            self.muhat = mean_array[k]
            close_list = self.simulate(length)
            open_list = [self.data.close[0], self.data.close[0]]
            open_list = open_list + close_list[1:-1]
            monte_carlo['open'] = open_list
            monte_carlo['close'] = close_list           
            wicks = self.simulate_wicks(length)
            close_great_equal_open = monte_carlo['close'] >= monte_carlo['open'] 
            monte_carlo.loc[close_great_equal_open, 'high'] =  ( (1+wicks['upper_green']) * monte_carlo['close'] * monte_carlo['close'] ) / monte_carlo['open']
            monte_carlo.loc[close_great_equal_open, 'low'] = ( monte_carlo['open'] * monte_carlo['open'] ) / ( (1+wicks['lower_green']) * monte_carlo['close'] )
            monte_carlo.loc[~close_great_equal_open, 'high'] = ( (1+wicks['upper_red']) * monte_carlo['open'] * monte_carlo['open'] ) / monte_carlo['close']
            monte_carlo.loc[~close_great_equal_open, 'low'] = ( monte_carlo['close'] * monte_carlo['close'] ) / ( (1+wicks['lower_red']) * monte_carlo['open'] )
            #Control mechas
            monte_carlo = PriceWorker.wicks_temporal_fix(monte_carlo)
            if volume:
                monte_carlo['volume'] = self.volume_simulator.simulated_volumen_from_candlestick_range(monte_carlo['high']-monte_carlo['low'],self.volumen_simulator_parameters)
            list_simulations.append(monte_carlo)                                
        return list_simulations    
    

class SimulateGeometricOHLC(GeometricBrownianMotionSimulator, WicksSimulator, VolumeSimulator, PriceWorker):
    def __init__(self, data, periods):
        self.data = data
        self.periods = periods
        WicksSimulator.__init__(self, self.data)
        self.mu, self.sigma = PriceWorker.calculate_mean_and_std(self, data.close)
        GeometricBrownianMotionSimulator.__init__(self, self.data.close[0], 1/self.periods, self.mu, self.sigma)
        self.volume_simulator = VolumeSimulator(norm,["loc","scale"],{"loc":self.lineal})
        lenght_data = len(self.data)
        bins = 2
        data_lenght_in_one_bin = 10000
        while data_lenght_in_one_bin<150:
            bins += 1
            data_lenght_in_one_bin = lenght_data/(bins+1)
        self.volumen_simulator_parameters = self.volume_simulator.get_parameters_volume_simulation(self.data,nbins=bins)
        
    def lineal(self,x,c1,x0):
        return c1*(x-x0)
        
    def simulate_ohlc_n_times(self, number_simulations, length, volume=False):
        '''Crea una tabla OHLC'''
        list_simulations = []       
        for k in range(number_simulations): 
            monte_carlo = pd.DataFrame()           
            close_list = self.simulate(length)
            open_list = [self.data.close[0], self.data.close[0]]
            open_list = open_list + close_list[1:-1]
            monte_carlo['open'] = open_list
            monte_carlo['close'] = close_list           
            wicks = self.simulate_wicks(length)
            close_great_equal_open = monte_carlo['close'] >= monte_carlo['open'] 
            monte_carlo.loc[close_great_equal_open, 'high'] =  ( (1+wicks['upper_green']) * monte_carlo['close'] * monte_carlo['close'] ) / monte_carlo['open']
            monte_carlo.loc[close_great_equal_open, 'low'] = ( monte_carlo['open'] * monte_carlo['open'] ) / ( (1+wicks['lower_green']) * monte_carlo['close'] )
            monte_carlo.loc[~close_great_equal_open, 'high'] = ( (1+wicks['upper_red']) * monte_carlo['open'] * monte_carlo['open'] ) / monte_carlo['close']
            monte_carlo.loc[~close_great_equal_open, 'low'] = ( monte_carlo['close'] * monte_carlo['close'] ) / ( (1+wicks['lower_red']) * monte_carlo['open'] )
            #Control mechas
            monte_carlo = PriceWorker.wicks_temporal_fix(monte_carlo)
            if volume:
                monte_carlo['volume'] = self.volume_simulator.simulated_volumen_from_candlestick_range(monte_carlo['high']-monte_carlo['low'],self.volumen_simulator_parameters)
            list_simulations.append(monte_carlo)    
        return list_simulations
    
    def simulate_ohlc_n_times_with_variable_mean(self, mean_array, number_simulations, length, volume=False):
        if len(mean_array) != number_simulations:
            print('No hay misma cantidad de medias y simulaciones')
            return
        '''Crea una tabla OHLC'''
        list_simulations = []       
        for k in range(number_simulations): 
            monte_carlo = pd.DataFrame()           
            self.mu = mean_array[k]
            close_list = self.simulate(length)
            open_list = [self.data.close[0], self.data.close[0]]
            open_list = open_list + close_list[1:-1]
            monte_carlo['open'] = open_list
            monte_carlo['close'] = close_list           
            wicks = self.simulate_wicks(length)
            close_great_equal_open = monte_carlo['close'] >= monte_carlo['open'] 
            monte_carlo.loc[close_great_equal_open, 'high'] =  ( (1+wicks['upper_green']) * monte_carlo['close'] * monte_carlo['close'] ) / monte_carlo['open']
            monte_carlo.loc[close_great_equal_open, 'low'] = ( monte_carlo['open'] * monte_carlo['open'] ) / ( (1+wicks['lower_green']) * monte_carlo['close'] )
            monte_carlo.loc[~close_great_equal_open, 'high'] = ( (1+wicks['upper_red']) * monte_carlo['open'] * monte_carlo['open'] ) / monte_carlo['close']
            monte_carlo.loc[~close_great_equal_open, 'low'] = ( monte_carlo['close'] * monte_carlo['close'] ) / ( (1+wicks['lower_red']) * monte_carlo['open'] )
            #Control mechas
            monte_carlo = PriceWorker.wicks_temporal_fix(monte_carlo)
            if volume:
                monte_carlo['volume'] = self.volume_simulator.simulated_volumen_from_candlestick_range(monte_carlo['high']-monte_carlo['low'],self.volumen_simulator_parameters)
            list_simulations.append(monte_carlo)    
        return list_simulations
    
class SimulateMLMertonOHLC(MertonJumpDiffusionSimulator, PriceWorker):
    
    def __init__(self, data, periods):
        self.data = data
        self.periods = periods
        self.central_process, self.outliers = PriceWorker.filter_carterafigueroa(self, self.data.close)
        self.muhat, self.sigmahat, self.Lambdahat, self.mu_jhat, self.sigma_jhat = PriceWorker.calculate_parameters_mertonjumpdiffusion_aproximation(self, self.central_process, self.outliers, self.periods)
        MertonJumpDiffusionSimulator.__init__(self, self.data.close[0], 1/self.periods, self.muhat, self.sigmahat, self.Lambdahat, self.mu_jhat, self.sigma_jhat)
        self.train()
        self.corr_mean=0
        
    def train(self):
        lecturas = 5
        self.hdata = self.data.copy()
        self.hdata["returns"]=self.hdata["close"]/self.hdata["open"]-1
        self.hdata["returns_shift"]=(self.hdata["close"]/self.hdata["open"]-1).shift(-1)
        self.hdata["upper"] = (self.hdata["high"] / np.maximum(self.hdata["open"],self.hdata["close"]))
        self.hdata["lower"] = (np.minimum(self.hdata["open"],self.hdata["close"]) / self.hdata["low"]) 
        self.hdata["log_volume"] = np.log(self.hdata["volume"])
        self.hdata["log_volume_pct"] = np.log(self.hdata["volume"]).pct_change()
        self.hdata = self.hdata[["returns_shift","log_volume_pct","upper","lower","log_volume"]].values[1:-1]
        
        X_train = []
        y_train = []
        for i in (range(lecturas, len(self.hdata))):
            X_train.append(self.hdata[i-lecturas:i, :].reshape(-1))
            y_train.append(self.hdata[i, 2:].reshape(-1))
        X_train, y_train = np.array(X_train), np.array(y_train)
        self.regressor= DecisionTreeRegressor()
        self.regressor.fit(X_train,y_train)
        self.start_data = X_train[-1].reshape(1,-1)
       
    def generate_upperlowerlogvolume(self,sim_returns):
        y_final = np.array([])
        X_sim = self.start_data
        for sim_return in sim_returns:
            X_sim = X_sim.reshape(1,-1)
            y_sim = self.regressor.predict(X_sim)
            y_sim = y_sim.reshape(-1)
            y_sim = np.append((y_sim[-1]/X_sim[0,-1])-1,y_sim)
            y_sim = np.append(sim_return,y_sim)
            X_sim = np.append(X_sim,y_sim)
            X_sim = X_sim[5:]
            y_final = np.concatenate((y_final,y_sim))
        y_final = y_final.reshape(-1,5)
        return y_final
    
    def force_correlation_with_previous_time(self,returns):
        last_real_return = self.data["close"].pct_change()[-1]
        c_t=1/(1+self.corr_mean)
        c_t_1=self.corr_mean/(1+self.corr_mean)
        returns = np.append(
            c_t*returns[0]+c_t_1*last_real_return
            ,c_t*returns[1:]+c_t_1*np.roll(returns,1)[1:])
        return returns
    
    def simulate_ohlc_n_times_with_variable_mean(self, mean_array, number_simulations, length, volume=False):
        if len(mean_array) != number_simulations:
            print('No hay misma cantidad de medias y simulaciones')
            return
        '''Crea una lista de tablas OHLC'''
        list_simulations = []     
        for k in range(number_simulations):
            monte_carlo = pd.DataFrame() 
            self.muhat = mean_array[k]
            self.mu_jhat = mean_array[k]
            price_list = np.array(self.simulate(length+1))
            returns = np.diff(price_list)/price_list[:-1]
            if self.corr_mean != 0:
                returns = self.force_correlation_with_previous_time(returns)
            close_prices = self.data.close[-1]*(1+returns).cumprod()
            open_prices = np.append(self.data.close[-1],close_prices[:-1])
            upperlowerlogvolume = self.generate_upperlowerlogvolume(returns)
            monte_carlo['open'] = open_prices
            monte_carlo['close'] = close_prices
            monte_carlo["volume"] = np.exp(upperlowerlogvolume[:,-1])
            high_prices =(upperlowerlogvolume[:,-3])*(np.maximum(open_prices,close_prices))
            low_prices = (np.minimum(open_prices,close_prices))/((upperlowerlogvolume[:,-2]))
            monte_carlo["high"] = high_prices
            monte_carlo["low"] = low_prices
            list_simulations.append(monte_carlo)                                
        return list_simulations 
    
    def fix_mean_corrfactor(self,historical_data):
        self.corr_mean = historical_data["close"].pct_change()[2:].corr(historical_data["close"].pct_change().shift(1)[2:])
        
class SimulateMLGeometricOHLC(GeometricBrownianMotionSimulator, PriceWorker):
    def __init__(self, data, periods):
        self.data = data
        self.periods = periods
        self.mu, self.sigma = PriceWorker.calculate_mean_and_std(self, data.close)
        self.mu, self.sigma = self.mu*self.periods,self.sigma*np.sqrt(self.periods)
        GeometricBrownianMotionSimulator.__init__(self, self.data.close[0], 1/self.periods, self.mu, self.sigma)
        self.train()
        self.corr_mean=0
        
    def train(self):
        lecturas = 5
        self.hdata = self.data.copy()
        self.hdata["returns"]=self.hdata["close"]/self.hdata["open"]-1
        self.hdata["returns_shift"]=(self.hdata["close"]/self.hdata["open"]-1).shift(-1)
        self.hdata["upper"] = (self.hdata["high"] / np.maximum(self.hdata["open"],self.hdata["close"]))# /
        self.hdata["lower"] = (np.minimum(self.hdata["open"],self.hdata["close"]) / self.hdata["low"]) #/
        self.hdata["log_volume"] = np.log(self.hdata["volume"])
        self.hdata["log_volume_pct"] = np.log(self.hdata["volume"]).pct_change()
        self.hdata = self.hdata[["returns_shift","log_volume_pct","upper","lower","log_volume"]].values[1:-1]
        X_train = []
        y_train = []
        for i in (range(lecturas, len(self.hdata))):
            X_train.append(self.hdata[i-lecturas:i, :].reshape(-1))
            y_train.append(self.hdata[i, 2:].reshape(-1))
        X_train, y_train = np.array(X_train), np.array(y_train)
        self.regressor= DecisionTreeRegressor()
        self.regressor.fit(X_train,y_train)
        self.start_data = X_train[-1].reshape(1,-1)
       
    def generate_upperlowerlogvolume(self,sim_returns):
        y_final = np.array([])
        X_sim = self.start_data
        for sim_return in sim_returns:
            X_sim = X_sim.reshape(1,-1)
            y_sim = self.regressor.predict(X_sim)
            y_sim = y_sim.reshape(-1)
            y_sim = np.append((y_sim[-1]/X_sim[0,-1])-1,y_sim)
            y_sim = np.append(sim_return,y_sim)
            X_sim = np.append(X_sim,y_sim)
            X_sim = X_sim[5:]
            y_final = np.concatenate((y_final,y_sim))
        y_final = y_final.reshape(-1,5)
        return y_final
    
    def force_correlation_with_previous_time(self,returns):
        last_real_return = self.data["close"].pct_change()[-1]
        c_t=1/(1+self.corr_mean)
        c_t_1=self.corr_mean/(1+self.corr_mean)
        returns = np.append(
            c_t*returns[0]+c_t_1*last_real_return
            ,c_t*returns[1:]+c_t_1*np.roll(returns,1)[1:])
        return returns
    
    def simulate_ohlc_n_times_with_variable_mean(self, mean_array, number_simulations, length, volume=False):
        if len(mean_array) != number_simulations:
            print('No hay misma cantidad de medias y simulaciones')
            return
        '''Crea una lista de tablas OHLC'''
        list_simulations = []     
        for k in range(number_simulations):
            monte_carlo = pd.DataFrame() 
            self.mu = mean_array[k]
            price_list = np.array(self.simulate(length+1))
            returns = np.diff(price_list)/price_list[:-1]
            if self.corr_mean != 0:
                returns = self.force_correlation_with_previous_time(returns)
            close_prices = self.data.close[-1]*(1+returns).cumprod()
            open_prices = np.append(self.data.close[-1],close_prices[:-1])
            upperlowerlogvolume = self.generate_upperlowerlogvolume(returns)
            monte_carlo['open'] = open_prices
            monte_carlo['close'] = close_prices
            monte_carlo["volume"] = np.exp(upperlowerlogvolume[:,-1])
            high_prices =(upperlowerlogvolume[:,-3])*(np.maximum(open_prices,close_prices))
            low_prices = (np.minimum(open_prices,close_prices))/((upperlowerlogvolume[:,-2]))
            monte_carlo["high"] = high_prices
            monte_carlo["low"] = low_prices
            list_simulations.append(monte_carlo)                                
        return list_simulations 
    
    def fix_mean_corrfactor(self,historical_data):
        self.corr_mean = historical_data["close"].pct_change()[2:].corr(historical_data["close"].pct_change().shift(1)[2:])
    
    
class FillNa:
                
    def fill_serie(self, serie_price, seed=None):
        """
        input:
            - serie_price: Serie temporal con NaN. 
                Importante: La serie no debe contener nan en los extremos
            - seed (default=None): semilla aleatoria
        output:
            - Serie completa
        """        
        #fijo semilla aleatoria para tratar siempre con la misma serie    
        np.random.seed(seed)
        #Detecto nans
        nans_index = []    
        for i in range(len(serie_price)):
            if np.isnan(serie_price[i]):
                nans_index.append(i)
        
        #Separo lista de nans en sus ventanas temporales
        nans_windows = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(nans_index), lambda x: x[0]-x[1])]
        
        #Relleno los nans con el comportamiento real de los precios: elemento deterministico + elemento estocastico
        for window in nans_windows:
            start_gap = window[0] - 1
            end_gap = window[-1] + 1
            path_missing = serie_price[end_gap] - serie_price[start_gap]
            
            #Este es el incremento lineal para unir los puntos
            drift = path_missing / ( len(window) + 1 )
            
            #Pero los retornos no serian los correctos!! Serian suaves y siempre en la misma direccion
            #Por esto mismo le agrego un shock aleatorio siguiendo la distribucion historica de
            #los retornos
            distribution_mean = 0#serie_price[start_gap-12*7:start_gap].pct_change().mean()
            distribution_std = serie_price[start_gap-12*7:start_gap].pct_change().std()
            
            for x in range(start_gap+1, end_gap):
                noise = np.random.normal(distribution_mean, distribution_std)
                serie_price[x] = (serie_price[x-1] + drift*(1 + noise) ) 
        
        return serie_price
    
    def fill_ohlc(self, df_price):
        """
        Completa Dataframe de precios OLHC (puede o no incluir volumen) simulando los precios faltantes 
        input:
            - df_price: DataFrame con precios olhc (incluye volumen).
            - seed (default=None): Semilla aleatoria
        output:
            - DataFrame con nan completos
        """
        #close fillna
        df_price["close"]= self.fill_serie(df_price["close"])
        #open fillna
        for i,row in df_price[df_price["open"].isna()]["open"].iteritems():
            df_price.loc[i,"open"]=df_price["close"].shift(1).loc[i]   
        #high/low fillna
        df_price["low"]=self.fill_serie(df_price["low"])
        df_price["high"]=self.fill_serie(df_price["high"])
        max_close_open=np.maximum(df_price["close"],df_price["open"])
        min_close_open=np.minimum(df_price["close"],df_price["open"])
        df_price["low"]=np.where(df_price["low"]>min_close_open,min_close_open,df_price["low"])
        df_price["high"]=np.where(df_price["high"]<max_close_open,max_close_open,df_price["high"])
        #volume fillna
        volume_simulator = VolumeSimulator(norm,["loc","scale"],{"loc":self.lineal})
        lenght_data = len(df_price.dropna())
        bins = 2
        data_lenght_in_one_bin = 10000
        while data_lenght_in_one_bin<150:
            bins += 1#https://gitlab.com/xcapit/investigacion/xcapit_util.git
            data_lenght_in_one_bin = lenght_data/(bins+1)
        volumen_simulator_parameters = volume_simulator.get_parameters_volume_simulation(df_price.dropna(),nbins=bins)
        df_price['volume'] = np.where(np.isnan(df_price['volume']),
                   volume_simulator.simulated_volumen_from_candlestick_range(df_price['high']-df_price['low'],volumen_simulator_parameters),
                   df_price['volume'])
        return df_price
    
    def lineal(self,x,c1,x0):
        return c1*(x-x0)
    
    
class Fitter():
    
    def fit_normal(self, data):
        mean, std = norm.fit(data)
        return mean, std
    
    def fit_tstudent(self, data):
        df, loc, scale = t.fit(data)
        return df, loc, scale
    
    
class DistributionGenerator():
    
    def generate_normal(self, mean, std, size):
        return norm.rvs(mean, std, size=size)
    
    def generate_normal_restricted_by_std(self, mean, std, size, numberofstd):
        sample = np.empty(0)
        count = 0
        while count < size:
            random = self.generate_normal(mean, std, size=1)
            if (random >= mean-numberofstd*std) and (random <= mean+numberofstd*std):
                sample = np.append(sample, random)
                count += 1
        return sample
    
    def generate_tstudent(self, df, loc, scale, size):
        return t.rvs(df=df, loc=loc, scale=scale, size=size)
    
    def generate_tstudent_restricted_by_std(self, df, loc, scale, size, numberofstd):
        sample = np.empty(0)
        count = 0
        while count < size:
            random = self.generate_tstudent(df, loc, scale, size=1)
            if (random >= loc-numberofstd*scale) and (random <= loc+numberofstd*scale):
                sample = np.append(sample, random)
                count += 1
        return sample
    
    
                
                
            
    