from binance import Client
import pandas as pd
import sqlite3

def update_db(client):
    klines_btc_30MIN = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_30MINUTE, "1 JAN, 2024")
    klines_btc = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_4HOUR, "1 JAN, 2024")
    klines_eth_30MIN = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_30MINUTE, "1 JAN, 2024")
    klines_eth = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_4HOUR, "1 JAN, 2024")
    klines_bnb_30MIN = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_30MINUTE, "1 JAN, 2024")
    klines_bnb = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_4HOUR, "1 JAN, 2024")
    klines_bnb = pd.DataFrame(data=klines_bnb).iloc[:-1]

    klines_bnb.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_bnb = klines_bnb.astype(float)

    klines_eth = pd.DataFrame(data=klines_eth).iloc[:-1]
    klines_eth.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_eth = klines_eth.astype(float)

    klines_btc = pd.DataFrame(data=klines_btc).iloc[:-1]
    klines_btc.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_btc = klines_btc.astype(float)

    klines_bnb_30MIN = pd.DataFrame(data=klines_bnb_30MIN).iloc[:-1]
    klines_bnb_30MIN.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_bnb_30MIN = klines_bnb_30MIN.astype(float)

    klines_btc_30MIN = pd.DataFrame(data=klines_btc_30MIN).iloc[:-1]
    klines_btc_30MIN.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_btc_30MIN = klines_btc_30MIN.astype(float)

    klines_eth_30MIN = pd.DataFrame(data=klines_eth_30MIN).iloc[:-1]
    klines_eth_30MIN.columns = ['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    klines_eth_30MIN = klines_eth_30MIN.astype(float)

    conn = sqlite3.connect("sqlite.db")
    cursor  = conn.cursor()
    BTCUSDT_30MIN_max_time = cursor.execute("SELECT MAX(Open_time) FROM BTCUSDT_30MIN").fetchall()[0][0]
    BTCUSDT_4H_max_time = cursor.execute("SELECT MAX(Open_time) FROM BTCUSDT_4H").fetchall()[0][0]
    ETHUSDT_30MIN_max_time = cursor.execute("SELECT MAX(Open_time) FROM ETHUSDT_30MIN").fetchall()[0][0]
    ETHUSDT_4H_max_time = cursor.execute("SELECT MAX(Open_time) FROM ETHUSDT_4H").fetchall()[0][0]
    BNBUSDT_30MIN_max_time = cursor.execute("SELECT MAX(Open_time) FROM BNBUSDT_30MIN").fetchall()[0][0]
    BNBUSDT_4H_max_time = cursor.execute("SELECT MAX(Open_time) FROM BNBUSDT_4H").fetchall()[0][0]

    klines_btc_30MIN = klines_btc_30MIN.loc[klines_btc_30MIN.Open_time > BTCUSDT_30MIN_max_time]
    klines_btc = klines_btc.loc[klines_btc.Open_time > BTCUSDT_4H_max_time]
    klines_eth_30MIN = klines_eth_30MIN.loc[klines_eth_30MIN.Open_time > ETHUSDT_30MIN_max_time]
    klines_eth = klines_eth.loc[klines_eth.Open_time > ETHUSDT_4H_max_time]
    klines_bnb_30MIN = klines_bnb_30MIN.loc[klines_bnb_30MIN.Open_time > BNBUSDT_30MIN_max_time]
    klines_bnb = klines_bnb.loc[klines_bnb.Open_time > BNBUSDT_4H_max_time]

    klines_btc_30MIN.to_sql("BTCUSDT_30MIN", con=conn, if_exists='append', index=False)
    klines_btc.to_sql("BTCUSDT_4H", con=conn, if_exists='append', index=False)
    klines_eth_30MIN.to_sql("ETHUSDT_30MIN", con=conn, if_exists='append', index=False)
    klines_eth.to_sql("ETHUSDT_4H", con=conn, if_exists='append', index=False)
    klines_bnb_30MIN.to_sql("BNBUSDT_30MIN", con=conn, if_exists='append', index=False)
    klines_bnb.to_sql("BNBUSDT_4H", con=conn, if_exists='append', index=False)

    cursor.close()
    conn.close()