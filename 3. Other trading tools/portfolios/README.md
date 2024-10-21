# template_investigacion

Para ejecutar la exploración de parámetros y muestreo ejecutar run_optimizacion.py

run_optimizacion.py realiza un loop sobre la funcion main definida en main_optimizacion.py. Esta función utiliza la clase ModelValidation (rama dev) para realizar la búsqueda de parámetros. Para lo anterior definió un función a optimizar "ObjectiveFunction" que se encuentra en src/Function_Portfolio.py.

######ObjectiveFunction
Esta clase además de contener los métodos estándar necesarios para la ejecución en ModelOptimization, también cuenta con un dataframe donde las métricas de cada uno de los modelos que se generaron en el código (en regiones baseline, entrenamiento, validación y test)
