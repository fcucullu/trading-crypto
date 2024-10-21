import pandas as pd
from datetime import datetime
from influxdb import DataFrameClient


class CandlestickRepository:
    TABLE_WITH_1MINUTE_CANDLES = 'candles'
    PREPROD_DB_NAME = 'xcapit'
    PREPROD_HOST = '10.0.51.11'
    PREPROD_PORT = 8086
    
    def __init__(self, db_name, host, port):
        self.db_name = db_name
        self.influx_client = DataFrameClient(host, port)
    
    @classmethod
    def preprod_repository(cls):
        return cls(cls.PREPROD_DB_NAME, cls.PREPROD_HOST, cls.PREPROD_PORT)
        
    def get_one_minute_candlestick(
        self, 
        pair: str, 
        exchange: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
    
        """Function gets one minute candles data for `pair` from influxdb. 
        
        Arguments:
            pair {str} -- the pair of the data to request
            exchange {str} -- exchange used as source
            start_time {datetime} -- when the time interval starts
            end_time {datetime} -- when the time interval ends
        
        Returns:
            pd.DataFrame -- DataFrame with candles data.
        """
    
        query = f"""select * from {table} where 
        exchange=$exchange and pair=$pair 
        and $start_time <= time and time<$end_time"""

        bind_params = {
            'exchange': exchange,
            'pair': pair,
            'start_time': f"{start_time}".split('+')[0],
            'end_time': f"{end_time}".split('+')[0]
        }
        result = self.influx_client.query(
            query,
            bind_params=bind_params,
            database=self.db_name
        )
        return result.get(self.TABLE_WITH_1MINUTE_CANDLES, pd.DataFrame())
    
    def get_candlestick(
        self, 
        pair: str, 
        exchange: str, 
        size: int,
        start_time: datetime, 
        end_time: datetime,
    ) -> pd.DataFrame:
        """Function gets candles data for `pair` from influxdb with candles of size `size` minutes. 
        
        Arguments:
            pair {str} -- the pair of the data to request
            exchange {str} -- exchange used as source
            size {int} -- minutes used to construct the candles 
            start_time {datetime} -- when the time interval starts
            end_time {datetime} -- when the time interval ends
        
        Returns:
            pd.DataFrame -- DataFrame with candles data.
        """

        
        query = f"""select \
        first(open) AS open, last(close) AS close, max(high) AS high, min(low) as low, sum(volume) as volume\
        from candles where exchange=$exchange and pair=$pair and time>=$start_time and time<$end_time 
        GROUP BY time({size}m)""" 

        bind_params = {
            'exchange': exchange,
            'pair': pair,
            'start_time': f"{start_time}".split('+')[0],
            'end_time': f"{end_time}".split('+')[0]

        }
        result = self.influx_client.query(
            query,
            bind_params=bind_params,
            database=self.db_name
        )
        candlesticks = result.get(self.TABLE_WITH_1MINUTE_CANDLES, pd.DataFrame())
        candlesticks['exchange'] = exchange
        candlesticks['pair'] = pair
        
        return candlesticks

