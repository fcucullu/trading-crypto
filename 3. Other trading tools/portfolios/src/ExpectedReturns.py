# -*- coding: utf-8 -*-
from abc import ABC,abstractclassmethod

    
class AbstractCovariances(ABC):
    @abstractclassmethod
    def get_covariances():
        pass
    
    
class AbstractExpectedReturns(ABC):
    @abstractclassmethod
    def get_expected_returns():
        pass


class Covariances(AbstractCovariances):
    
    def __init__(self,prices,lecturas,ewm=False,**kwargs):
        
        if ewm==False:
            self.covariances=prices.pct_change().rolling(lecturas).cov()
        else:
            self.covariances=prices.pct_change().ewm(span=lecturas,**kwargs).cov()
        
    def get_covariances(self):
        
        return self.covariances.dropna()
        


class ExpectedReturns_Mean(AbstractExpectedReturns):
    
    def __init__(self,prices,lecturas,last_return_weight):
        
        previous_prices=(prices.shift(1)+prices.shift(2))/2.
        
        self.last_returns=(prices-previous_prices)/prices
        
        self.mean_returns=prices.pct_change().rolling(lecturas).mean()
        
        self.returns=self.mean_returns+last_return_weight*self.last_returns
        
        self.returns=self.returns.dropna()
    
    def get_expected_returns(self):
        
        return self.returns
        
    
        
        
        