
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from ExpectedReturns import AbstractExpectedReturns

class ExpectedReturns_RNN(AbstractExpectedReturns):
    def __init__(self,n_input_variables,n_output_variables,lecturas,n_LTSM_layers=5,path_load_model=""):
        
        self.lecturas=lecturas
        self.n_output_variables=n_output_variables
        self.n_input_variables=n_input_variables
        
        # Initialising Scaler
        self.sc = MinMaxScaler(feature_range = (0, 1))
        
        # Initialising the RNN
        self.regressor = Sequential()
        
        # Adding the first LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (self.lecturas,n_input_variables)))
        self.regressor.add(Dropout(0.2))
        
        for i in range(n_LTSM_layers-1):
            # Adding a second LSTM layer and some Dropout regularisation
            self.regressor.add(LSTM(units = 200, return_sequences = True))
            self.regressor.add(Dropout(0.2))
        
        # Adding the output layer
        self.regressor.add(Dense(units = self.n_output_variables))
        
        # Compiling the RNN
        #self.regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.regressor.compile(optimizer = 'adam', loss = 'logcosh')

        if path_load_model!="":
            self.load(path_load_model)
        
        
        return
    
    def delete_nan(self,*X):
        
        nan_rows=[]
        for x in X:
            
            nanvec=np.argwhere(np.isnan(x))
            nan_index=[i[0] for i in nanvec]
            nan_rows+=list(set(nan_index))
        
        X_out=[]
        for x in X:
            X_out.append(np.delete(x,nan_rows,0))
            
        return X_out
    
    def train(self,prices,epochs = 100, batch_size = 100):
        
        prices_scaled = self.sc.fit_transform(prices.values)

        X_train = []
        y_train = []
        
        for i in (range(self.lecturas, len(prices_scaled))):
           
            X_train.append(prices_scaled[i-self.lecturas:i, :])
            y_train.append(prices_scaled[i, :self.n_output_variables])
                
        del i

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train,y_train=self.delete_nan(X_train,y_train)
        print(X_train.shape,y_train.shape)
        
        self.regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

        return
    
    def save(self,file):
        self.regressor.save("model.h5")
        
    def load(self,file):

        self.regressor = load_model(file)

        return 
    
    def get_predicted_prices(self,prices,stable="USDT"):
        
        prices_columns=prices.columns        
        
        prices_scaled=self.sc.transform(prices.values)
        X_test = []
        for i in range(self.lecturas, len(prices_scaled)):
            X_test.append(prices_scaled[i-self.lecturas:i, :])
        
        X_test = np.array(X_test)
        #X_test=self.delete_nan(X_test)
        
        predicted_prices = self.regressor.predict(X_test)
        zeros=np.zeros((len(prices_scaled)-self.lecturas,self.n_input_variables-self.n_output_variables))
        predicted_prices = np.append(predicted_prices,zeros,axis=1)

        predicted_prices = self.sc.inverse_transform(predicted_prices)

        predicted_prices=predicted_prices[:,:self.n_output_variables]
        
        predicted_prices=pd.DataFrame(predicted_prices,columns=prices_columns[:self.n_output_variables],index=prices.index[self.lecturas:])
        
        predicted_prices[stable]=1

        self.predicted_prices=predicted_prices

        prices=pd.DataFrame(prices,columns=prices_columns[:self.n_output_variables],index=prices.index[self.lecturas:])
                
        prices[stable]=1


        self.prices=prices
        
        
    def get_expected_returns(self):
        
        
        scale=(self.prices/self.predicted_prices).mean()
        output=((scale*self.predicted_prices-self.prices)/self.prices).dropna()
        
        
        return output
    
    