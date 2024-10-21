# template_investigacion

        La funcion detecta cuando el volumen de las obs_acumuladas sobrepasa una 
        media movil y calcula el retorno acumulado que se obtendria en la ventana 
        objetivo. La ventana objetivo se define desde el OPEN del dia siguiente a
        la senial hasta el CLOSE del final de la ventana.
            Ejemplo:
                Ventana_objetivo = 2 >>> Retorno = Close[2+1] / Open[1] - 1
                Ventana_objetivo = 0 >>> Retorno = Close[0+1] / Open[1] - 1
        
        La direccion de la operacion se determina por la banda de bollinger
        que rompe el precio.
        
        inputs:
            df = Data financiera con el dato mas reciente al final
            obs_acumuladas = observaciones hacia atras para acumular volumen
            ventana_objetivo = cantidad de dias hacia adelante a analizar
            media = periodos de la media movil de volumen
            tipo_media = tipo de media movil. Acepta 'EMA' y 'MM'
            tipo_retorno = tipo del retorno futuro calculado. Acepta 'ABS' y 'NOABS'.
                El primero es el retorno absoluto de la ventana objetivo y el segundo
                es el retorno direccional segun la direccion de las ultimas obs_acumuladas.
            periodo_bandas = es la cantidad de observaciones hacia atras para calcular las bandas.
            desviacion_bandas = es la cantidad de desvios estandar que el CLOSE debe superar 
                para calcular la direccion de la serie.
        
        output:
            Dataframe con las siguientes columnas
                'media': media de volumen analizada
                'tipo_media': tipo de media analizada
                'tipo_retorno': tipo de retorno analizado
                'obs_acumuladas: cantidad de observaciones en las que se acumulo el vol
                'ventana_objetivo': rango futuro en donde se maximiza retorno
                'trades': cantidad de seniales detectadas
                'amplitud_mediana': La amplitud mediana de las velas (high-low)
                'retorno_futuro': retorno medio futuro obtenido por la estrategia
                'perido_bandas': obs acumuladas para calcular bandas
                'desvio_bandas': cantidad de desvios para calcular direccion
                'ret_esperado': Es la metrica para definir que estrategia es mejor.
                        Es el retorno promedio dividido por la longitud de la data.
                        Asi alcanzo el retorno promedio por observacion que podria
                        alcanzar la estrategia. Luego lo anualizo * 365
        
