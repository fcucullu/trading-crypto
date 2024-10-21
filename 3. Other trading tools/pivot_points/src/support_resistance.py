import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class SupportResistance():
    def isSupport(self, df, i, n):
        if (((df.close[i] < df.close[i-n:i]).unique()).any() == True) or (((df.close[i] < df.close[i+1:i+1+n]).unique()).any() == True):
            return False
        else:
            return True if i>n else False
        
    def isResistance(self, df, i, n):
        if (((df.close[i] > df.close[i-n:i]).unique()).any() == True) or (((df.close[i] > df.close[i+1:i+1+n]).unique()).any() == True):
            return False
        else:
            return True if i>n else False
    
    def isFarFromLevel(self, l, s, levels):
        return np.sum([abs(l-x[2]) < s for x in levels]) == 0
    
    def isCloseFromResistance(self, df, levels):
        #1) que la diferencia entre el nivel importante y el close sea menor a X ATR
        #2) que el precio este por debajo del nivel, indicando ser resistencia long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] < x[2] 
                                        and row.name > x[0])
                                        for x in levels]
                                ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
        
    def isCloseFromSupport(self, df, levels):
        #1) que la diferencia entre el nivel importante y el close sea menor a un ATR
        #2) que el precio este por encima del nivel, indicando ser soporte long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] > x[2] 
                                        and row.name > x[0])
                                        for x in levels]
                                ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
    
    def combine_level(self, l, s, levels):
        obs, rep, prices = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in levels:
              if abs(l-x[2]) < s:
                  obs, rep, prices = np.append(obs, x[0]), np.append(rep, x[1]), np.append(prices, x[2])
        levels = [x for x in levels if x[0] not in obs] #Elimina los duplicados
        levels.append((obs.min(), rep.sum()+1, np.append(prices, l).mean()))
        return levels
        
    def find_important_levels(self, df, n):
        levels, pps = [], []
        for i in range(n,df.shape[0]-n):
            if self.isSupport(df,i, n):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromLevel(l, s, levels):
                    levels.append((df.index[i],0,l))
                else:
                    levels = self.combine_level(l, s, levels)
            elif self.isResistance(df,i, n):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromLevel(l, s, levels):
                    levels.append((df.index[i],0,l))
                else:
                    levels = self.combine_level(l, s, levels)
        levels = [i for i in levels if i[1] != 0 or i[0] > df.index[-168]]
        return levels, pps
                
    def find_important_areas(self, df, n):
        areas, pps = [], []
        for i in range(n,df.shape[0]-n):
            if self.isSupport(df,i, n):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromArea(l, s, areas):
                    areas.append((df.index[i],0,l,l))
                else:
                    areas = self.combine_area(l, s, areas)
            elif self.isResistance(df,i, n):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromArea(l, s, areas):
                    areas.append((df.index[i],0,l,l))
                else:
                    areas = self.combine_area(l, s, areas)
        areas = [i for i in areas if i[1] != 0 or i[0] > df.index[-168]]
        return areas, pps
    
    def combine_area(self, l, s, areas):
        obs, rep, maxs, mins = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in areas:
              if (abs(l-x[2]) < s or abs(l-x[3]) < s):
                  obs, rep, maxs, mins = np.append(obs, x[0]), np.append(rep, x[1]), np.append(maxs, x[2]), np.append(mins, x[3])
        areas = [x for x in areas if x[0] not in obs] #Elimina los duplicados
        areas.append((obs.min(), rep.sum()+1, np.append(maxs, l).max(), np.append(mins, l).min()))
        return areas
    
    def isFarFromArea(self, l, s, areas):
        return (np.sum([abs(l-x[2]) < s for x in areas]) == 0 or
                np.sum([abs(l-x[3]) < s for x in areas]) == 0)
    
    def isCloseFromLowerArea(self, df, areas):
        #1) que la diferencia entre el nivel importante y el close sea menor a X ATR
        #2) que el precio este por debajo del nivel, indicando ser resistencia long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[3] - row['close']) <= row['ATR']/4 
                                        and row['close'] < x[3] 
                                        and row.name > x[0])
                                        for x in areas]
                                    ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
        
    def isCloseFromUpperArea(self, df, areas):
        #1) que la diferencia entre el nivel importante y el close sea menor a un ATR
        #2) que el precio este por encima del nivel, indicando ser soporte long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] > x[2] 
                                        and row.name > x[0])
                                        for x in areas]
                                    ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
    
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
        df.loc[df.index[mean],'ATR'] = round( df.loc[df.index[1:mean+1],"TR"].rolling(window=mean).mean()[mean-1], 4)
        const_atr = (mean-1)/mean
        const_tr = 1/mean
        ATR=df["ATR"].values
        TR=df["TR"].values
        for index in range(mean+1, len(df)):
            ATR[index]=ATR[index-1]*const_atr+TR[index]*const_tr
            #df.loc[df.index[index],'ATR'] = (df.loc[df.index[index-1],'ATR'] * const_atr+ df.loc[df.index[index],'TR'] * const_tr) 
        df["ATR"]=ATR
        return df

    def plot_all(self, df, levels, pps):
        fig, ax = plt.subplots(figsize=(12,12))
        plt.plot(df.close, linewidth = 1, color = 'black')
        plt.legend('Price')
        plt.title('Price + PivotPoints')
        plt.xlabel('Observations')
        plt.ylabel('Price')
        for level in levels:
            plt.hlines(level[2],xmin=level[0],\
                    xmax=df.index[-1],colors='blue')
        for pp in pps:
            plt.scatter(pp[0], pp[1], s=50, color='r')
        fig.show()



sr = SupportResistance()
df = sr.calculate_average_true_range(df, 22)
n=10
areas, pps = sr.find_important_areas(df, n)     
df['close_support'] = sr.isCloseFromUpperArea(df, areas)
df['close_resistance'] = sr.isCloseFromLowerArea(df, areas)
fig, ax = plt.subplots(figsize=(15,15))
plt.plot(df.close, linewidth = 1, color = 'black')
plt.plot(df.ma_short, linewidth = 1, color = 'red')
plt.plot(df.ma_long, linewidth = 1, color = 'blue')
plt.legend('Price')
plt.title('Price + PivotPoints')
plt.xlabel('Observations')
plt.ylabel('Price')
for area in areas:
    ax.fill_between(df.index[df.index>area[0]], area[2], area[3], color='y')
for pp in pps:
    plt.scatter(pp[0], pp[1], s=50, color='b')
for row in range(len(df)):
    if df.iloc[row]['close_support']:
        plt.scatter(df.iloc[row].name, df.iloc[row].close, s=25, color='g')
    if df.iloc[row]['close_resistance']:
        plt.scatter(df.iloc[row].name, df.iloc[row].close, s=25, color='r')
fig.show()








