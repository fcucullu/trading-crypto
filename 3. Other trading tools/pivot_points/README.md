# Localización y cálculo de Puntos Pivote

En este .py vamos a localizar puntos de máximos y mínimos de las series, con el objetivo de ubicar puntos pivots en cada mínimo/máximo. Para ello, utilizaremos dos formas de encontrar los puntos extremos.

1. A partir de los precios de cierre.
Diremos que en un tiempo  𝑡  de la serie, el precio de cierre correspondiente es un mínimo (máximo) si para los  𝑛  tiempos anteriores y posteriores, los precios de cierre son mayores (menores).

2. A partir de una media simple corta.
Primero buscaremos como el en caso anterior, máximos y mínimos de una media corta (por ejemplo, una media simple de un día). Dado el momento  𝑡  donde se detecta un punto extremo (de la media), se buscará un punto extremo de la serie original trazando una ventana de  𝑛  periodos anteriores a  𝑡 .


