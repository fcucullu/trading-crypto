import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_normalize(df,col):
    '''
    Los parámetros de entrada son un df y una columna de ese df.
    La salida es el df donde la columna comienza con su valor en 100.
    '''
    df_norm = df[col].pct_change().replace(np.nan,0) + 1
    df_norm.iloc[0] = 100
    df_norm =  df_norm.cumprod()
    return df_norm

def get_medias(df, col, inicial_,final_, q, ema_short, ema_long, periodo_media,media = True ):
    '''
    Los parámetros de entrada son un df y una columna de ese df (usualmente precio), fechas inicial y final, 
    el número de periodos para las medias (q), la longuitud de las medias corta y larga.
    Obtenemos el dataframe graph que tiene como columnas, col (precio) y el valor de las medias móviles exponenciales corta y larga.
    True or False depende si queremos que nos sea mostrado el gráfico de precio y medias móviles seas mostrado.
    '''
    resultado = []
    resultado_graph = []
    df_aux = pd.DataFrame()
    
    df_aux = df
    column = df_aux[col] 

    low = column.index.get_loc(inicial_)
    high = column.index.get_loc(final_)
    pct_close = column.pct_change()[low:high]
    close = np.array(column)

    isize = [ema_short, ema_long] 
    arreglo = np.array(isize)

    ewma = {q*isize: np.array(column.ewm(span=q*isize, adjust =  False).mean())[low:high+1] for isize in arreglo}
    
    graph = pd.DataFrame()

    graph[col] = column.loc[inicial_:final_]#[:-1] 
    graph['ema_short'] = ewma[q*isize[0]]
    graph['ema_long'] = ewma[q*isize[1]]
    
    if media == True:
        graph['media_{}'.format(periodo_media)] = column.loc[inicial_ : final_][:-1].rolling(periodo_media).mean().replace(np.nan,0)
     
    graph.plot(figsize=(15,4))   
    return graph


def on_range(df,col, periodo, desviacion, inicial, final): #ndev = 1.75
    '''
    Los parámetros de entrada son un df y una columna de ese df.
    El periodo es cuantós periodos hacia atrás tomar.
    La desviacion es qué proporción de la desviación estandard tomar.
    Los parámetros inicial y final, son las fechas de comienzo y finalización.
    La función devuelve un df (df_copy) con los valores de las bandas superior e inferior, su diferencia y una tasa de cambio entre ellas. 
    '''
    df_copy = df.copy()
    df_copy = df[inicial:final]
    
    low = df[col].index.get_loc(inicial)
    high = df[col].index.get_loc(final)
    pct_close = df[col].pct_change()[low:high]
    close = np.array(df[col])
    
    df_copy['bb_high'] =  volatility.bollinger_hband(df_copy[col], n= periodo, ndev= desviacion, fillna=False)
    df_copy['bb_low'] =  volatility.bollinger_lband(df_copy[col], n= periodo, ndev= desviacion, fillna=False)
    df_copy['bb_dif'] = df_copy['bb_high'] - df_copy['bb_low']
    df_copy['bb_ret'] = (df_copy['bb_high'] - df_copy['bb_low'])/df_copy['bb_low'] 
       
    return df_copy

def get_df(df_):
    '''
    Como parámetro de entrada ingresamos un df, que debe tener las columnas 'close' y 'volume'.
    Esta función recibe dataframe obtenidos de datos de Binance y ajusta el nombre de las columnas y los índices.
    '''
    df = df_.copy()
    df = df.rename(columns= {'Unnamed: 0':'Date'}).set_index('close_time')
    df.index = pd.to_datetime(df.index)
    df.index = df.index + pd.Timedelta(1, unit="ms") 
    df.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in df.index])    
    df = df[['open','high','low','close','volume']]
    return df

def get_info(df,fecha_inicial,fecha_final,columns, long, short):
    '''
    Los parámetros de entrada son un df y columnas de ese df (close y volume), fechas inicial y final,
    y las longuitudes de los medias (simples).
    La salida es un df (media) donde tenemos el valor de close, y de las medias simples corta y larga.
    Además nos muestra el gráfico de las mismas. 
    '''
    media = pd.DataFrame()

    df = get_normalize(df[fecha_inicial:fecha_final], columns)

    media_long = df.close.rolling(long).mean().replace(np.nan,0).replace(0,df.close.iloc[0])
    media_short = df.close.rolling(short).mean().replace(np.nan,0).replace(0,df.close.iloc[0])

    media['close'] = df.close
    media['media_long'] = media_long
    media['media_short'] = media_short
    media[['close', 'media_long']].plot(figsize = (15,5))
    plt.grid(True)
    return media

def get_graphics_volume(df,fecha_inicial, fecha_final,periodo, fecha_clave, p):
    '''
    Los parámetros de entrada son un df (con una columna volumen), fechas de inicio y fin, un periodo hacia atras, una fecha 
    (fecha_clave) que indica hasta donde quiero graficar el volumen y un parámetro p (de ajuste).
    La salida es la media (simple) del volumen y un gráfico de medias y barras.
    '''
    low = df.index.get_loc(fecha_clave)
    df = df[fecha_inicial:fecha_final]
    
    media_vol = df.volume.rolling(periodo).mean().replace(np.nan,df.close.iloc[0])
    volumen = df.volume.replace(np.nan,0)
    volumen[:low+1].plot(figsize = (15,5))
    media_vol[:low+1].plot(figsize = (15,5))
    media_vol[:fecha_clave].tail()

    plt.figure(figsize = (15,5))
    vol_g  = volumen[low-p:low+1]
    plt.bar(range(len(vol_g)),  vol_g)
    plt.show()
    porcentaje = round(100*((df.volume.loc[fecha_clave] / media_vol.loc[fecha_clave]) - 1),4)
    print('La diferencia entre la media del volumen y el volumen en la fecha_clave es un {} %.'.format(porcentaje))
    
    return media_vol

def get_graphic_signals(df_precio,df_señal,col,color):
    '''
    Los parámetros de entrada son un df con los precios, otro df con las señas de compra (y venta), una columna del df, close y color 
    (modificar el color de las líneas punteadas)
    La salida es un gráfico, donde marca en qué momentos se compra (y vende).
    '''
    df_precio[col].plot(figsize = (15,4))
    for i in range(len(df_señal.index)):
        x = df_señal.index[i]
        plt.axvline(x , linewidth=0.5, color= color, ls = '--')
    plt.show()
    
def get_dates(df_abrir, df_cerrar):
    '''
    Se ingresan dos dataframes, uno de fechas donde se compraría y otro de fechas donde se vendería. 
    La función arroja una lista (fechas), donde cada elemento de la lista es un par [fecha_comprar, fecha_vender], que indica en que
    momento comprar y cuando vender dentro de esa ventana de tiempo. 
    '''

    fechas = list()

    fecha_inicial_c = df_abrir.index[0]
    if df_cerrar.loc[fecha_inicial_c:].empty == True:
        return fechas

    fecha_inicial_v = df_cerrar.loc[fecha_inicial_c:].index[0]

    fechas.append([fecha_inicial_c, fecha_inicial_v])

    while (fecha_inicial_v < df_cerrar.index[-1]) and (fecha_inicial_c < df_cerrar.index[-1]) and (fecha_inicial_v < df_abrir.index[-1]): 

        fecha_inicial_c = df_abrir.loc[fecha_inicial_v:].index[0] 
        if fecha_inicial_c < df_cerrar.index[-1]:

            fecha_inicial_v = df_cerrar.loc[fecha_inicial_c:].index[0]
            fechas.append([fecha_inicial_c, fecha_inicial_v]) 

    return fechas    

def get_estrategy(df, col, date, num_trades, delta, unit):
    '''
    Dado un df, una columna de df, una lista de fechas de compra y venta, el número de trades, y un delta y unidad, devuelve 
    un dataframe con la estrategia por haber comprado y vendido en las fechas indicadas. El delta es porque dada una recomendación de
    compra o venta, la acción se realiza un periodo después de dicha recomendación. (Hasta el momento detal =2, unit = 'h') 
    '''
    df_estrategia = df.copy() 
    df_estrategia['señal'] = 0

    for i in range(num_trades):
        delta = pd.Timedelta(delta, unit)
        df_estrategia['señal'].loc[pd.to_datetime(date[i][0]) + delta : pd.to_datetime(date[i][1]) + delta] = 1
        
    df_estrategia['pct_change'] = df_estrategia[col].pct_change().replace(np.nan,0)     
    df_estrategia['retorno'] = (df_estrategia['señal'] * df_estrategia['pct_change']).replace(-0,0)
        
    df_estrategia['señal_shift'] = df_estrategia.señal.shift(1).replace(np.nan,0) 
    df_estrategia['comision'] = np.where(df_estrategia.señal != df_estrategia.señal_shift, 0.001, 0)
    df_estrategia['retorno_total'] = (1 + df_estrategia.retorno)*(1- df_estrategia.comision)
    
    
    return df_estrategia     

def get_returns(df, col):
    '''
    Dado un dataframe y una columna de retornos (pct_change), calcula los retornos de la inversión si comenzamos con 100. 
    '''
    retornos = df[col]
    retornos = retornos + 1
    retornos.iloc[0] = 100
    retornos = retornos.cumprod()
    return retornos


def get_global_strategy(df_buy, df_sell,df,col, period = 2, unit = 'h'):
    '''
    Los parámetros de entrada son los dataframes de compra y venta (cuándo compro, cuándo vendo), el df inicial con los precios 
    (variable close), una columna: 'close'. Y parámetros period y unit, para definir que: cuando se nos de la señal de compra y venta, 
    realizaremos estas acciones en un periodo posterior. (Al estar trabajando con velas de dos horas, fijé period = 2 y unit= 'h')
    La salida es un gráfico del comportamiento de mi estrategia global y un dataframe (df_estrategia).
    '''
    global fechas
    fechas =  get_dates(df_buy,df_sell)
    if len(fechas) == 0:
        return []

    #get_graphic(df,'close',fechas)
    df_estrategia = get_estrategy(df, 'close',  fechas, len(fechas), period, unit)

    btc_norm = get_normalize(df_estrategia,col)
    retorno_parcial = get_returns(df_estrategia, 'retorno')

    plt.figure(figsize = (10,4))
    btc_norm.plot()
    #retorno_parcial.plot()
    retorno_ = df_estrategia.retorno_total
    retorno_.iloc[0] = 100
    retorno_ = retorno_.cumprod()
    retorno_.plot()

    plt.legend(['btc','Estrategia'])
    return df_estrategia
