# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize


class Recomendador:
    
    def generar_recomendacion(self,
                              expected_returns,
                              covariances,
                              riesgo,
                              periods=365*12,
                              comision=0.001,
                              ultima=True,
                              start_date='',
                              end_date='',
                              previous_weights=None
                              ):
        
        covariances=covariances*periods
        expected_returns=expected_returns*periods
        comision=comision*periods
        if start_date == '':
            fecha_inicio = max(expected_returns.index.min(),covariances.index.min()[0])
        else:
            fecha_inicio = datetime.strptime(start_date, "%Y-%m-%d") # Agregar  %H:%M:%S para cuando se pueda limitar por hora
        if end_date == '':
            fecha_fin = min(expected_returns.index.max(),covariances.index.max()[0])
        else:
            fecha_fin = datetime.strptime(end_date, "%Y-%m-%d")
        # Si las fechas no estan en el df levanto una excepcion
        if len(expected_returns.loc[fecha_inicio:fecha_fin]) == 0:
            raise ValueError('There is no data for those dates')
        ndim = len(expected_returns.columns)
        constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x[:ndim]) - 1.0})
        boundary = ()
        for i in range(0, ndim):
            boundary = (*boundary, (0, None))
        solutions = np.empty((0, ndim))
        # Comienza a generar la recomendacion
        if not previous_weights:
            previous_weights=[1/ndim for i in range(ndim)]
            
        for ind in expected_returns.index:
            # Crear vectores de retornos y matriz de covarianzas
            returns = np.array(expected_returns.loc[ind])
            covariance = np.matrix(covariances.loc[ind])
            previous_weights=previous_weights*(1+returns)
            previous_weights=previous_weights/sum(previous_weights)            
            # Definir la función a optimizar (como estamos minimizando, hay que usar -f(x))
            def portfolio_function(x):
                weight_changes = (np.where((x-previous_weights) < 0, 0, (x-previous_weights))).sum()
                #comision=0
                return float(- np.dot(returns, x)
                             + riesgo * np.dot(x, np.dot(covariance, x).T)
                             + comision * weight_changes)

            problem = optimize.minimize(portfolio_function,
                                        previous_weights,
                                        bounds=boundary,
                                        constraints=constraint, method="SLSQP"
                                        #options={'maxiter': 10000, 'ftol': 1e-05, 'iprint': 1,
                                         #        'disp': False, 'eps': 0.500000e-7}#1.4901161193847656e-08}
                                       )
            # Apilemos los vectores de soluciones por cada día
            solutions = np.append(solutions, np.array(
                problem.x.reshape(1, ndim)), axis=0)
            previous_weights = np.array(problem.x.reshape(1, ndim))[0, :]
        # Transformamos el vector de soluciones a un DataFrame
        weights = pd.DataFrame(solutions,
                               index=pd.to_datetime(
                                   expected_returns.index, unit='ms'),
                                   columns=list(expected_returns.columns))
        #weights.iloc[self.df_columns]
        weights = weights.apply(lambda x: x/weights.sum(axis=1))
        return weights

