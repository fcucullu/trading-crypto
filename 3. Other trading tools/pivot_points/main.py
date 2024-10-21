import ray
import itertools


ray.init(webui_host='0.0.0.0')


def main(symbol, intervalo, fecha):
  print(f'symbol: {symbol}, intervalo: {intervalo}, fecha: {fecha}')


@ray.remote
def execute_x(symbol, intervalo, fecha):
    main(symbol, intervalo, fecha)


symbols = ["ETHUSDT", "BTCUSDT"]
intervalos = ["15m", "2h"]
fecha = '20200425'


params = itertools.product(symbols, intervalos)


futures = [execute_x.remote(i[0], i[1], fecha) for i in params]
ray.get(futures)
