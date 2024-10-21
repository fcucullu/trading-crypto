#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:58:34 2020

@author: farduh
"""
import numpy as np
import pandas as pd


class XcapitRatioCalculator:
    
    def __init__(self):
        
        self.base_in_results=0
        self.base_out_results=0
        
    def get_xcapit_ratio(self,validation_results,benchmark_model,span,umbral=0.5):
        
        validation_results=self.calculate_xcapit_ratios(validation_results,benchmark_model,umbral=0.5)
        Xcapit_ratio=validation_results["Xcapit_ratio"].ewm(span=span).mean().iloc[-1]
        return Xcapit_ratio,validation_results
        
    
    def add_base_to_compare(self,
                            validation_results,
                             benchmark_model):
        
        validation_results["bm_train"]=0
        validation_results["bm_validation"]=0
        
        if benchmark_model==None:
            return validation_results
        
        for i,row in validation_results.iterrows():
            benchmark_model.set_dates(row["train_date_init"],row["train_date_final"])
            validation_results["bm_train"].loc[i]=benchmark_model.function_to_optimize()
            benchmark_model.set_dates(row["val_date_init"],row["val_date_final"])
            validation_results["bm_validation"].loc[i]=benchmark_model.function_to_optimize()
        
        return validation_results
        

    def calculate_xcapit_ratios(self,validation_results,benchmark_model,umbral=0.5,in_sample_name="train",out_sample_name="validation"):
        
        validation_results=self.add_base_to_compare(validation_results,benchmark_model)
        validation_results["Xcapit_ratio"]=0
        
        for i,row in validation_results.iterrows():
            validation_results["Xcapit_ratio"].loc[i]=self.xcapit_ratio(
                    row["train"]-row["bm_train"],
                    row["validation"]-row["bm_validation"]
                    ,umbral)
        
        return validation_results
        
        
            
    def xcapit_ratio(self,in_sample,out_sample,umbral=0.5):
                
        xcapit_ratio=in_sample/np.maximum((in_sample-out_sample)**2,umbral)
        return xcapit_ratio
        
    
    
    