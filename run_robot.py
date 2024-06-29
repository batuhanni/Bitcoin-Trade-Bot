#from predictorv2 import predict_signal
from predictor import predict_signal
from datetime import datetime
import time
from pipeline import update_db
from binance import Client
import warnings
import requests
from dotenv import load_dotenv
import os

load_dotenv()
warnings.simplefilter(action='ignore')

TOKEN = os.environ.get("TOKEN") #TELEGRAM TOKEN
chat_id = os.environ.get("CHAT_ID") #TELEGRAM CHAT ID

client = Client(os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")) #BINANCE CREDENTIALS HERE
client.ping()

class Bot:
    def __init__(self) -> None:
        print("Bot has been initiliazied.")
        self.test_mode = True
        self.open_positions = []
        self.turn = 0
        self.base_quantity = None
        self.sl = None
        self.tp = None

    def predict_current_4H(self):
        pred_time, pred = predict_signal(date=datetime.now(),return_pred_time=True)
        self.current_4H_signal = pred[0]
        self.pred_time = pred_time

    def confidence_levels(self,confs):
        self.confs = confs

    def is_active(self):
        self.active = len(self.open_positions) >= 1

    def log_and_send_message(self):
        message = f"""**BOT Çalıştı.**
        {self.pred_time.strftime('%m/%d : %H')} --> {self.current_4H_signal}
        Açık işlemler: {self.open_positions} , Tur: {self.turn}
        *** Test Mode : {self.test_mode}"""
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)

    def create_order(self,side):
        
        client.futures_create_order(symbol='BTCUSDT', side=side, type='MARKET', quantity=self.base_quantity)
        last_price = float(client.get_ticker(symbol="BTCUSDT")['lastPrice'])
        
        if side=='BUY':
            stop_price = int((1 - self.sl) * last_price)
            take_price = int((1 + self.tp) * last_price)
        elif side=='SELL':
            stop_price = int((1 + self.sl) * last_price)
            take_price = int((1 - self.tp) * last_price)

        reverse_side = 'SELL' if side == 'BUY' else 'BUY'
        client.futures_create_order(symbol='BTCUSDT', side=reverse_side, type='STOP_MARKET', stopPrice=stop_price, quantity=self.base_quantity)
        client.futures_create_order(symbol='BTCUSDT', side=reverse_side, type='TAKE_PROFIT_MARKET', stopPrice=take_price, quantity=self.base_quantity)

    def close_order(self,side):
        client.futures_create_order(symbol='BTCUSDT', side=side, type='MARKET', quantity=self.base_quantity)

    def start(self):
        print("Bot has been started.")
        while 1:
            if (datetime.utcnow().timestamp() - datetime(2024,1,1,0,0).timestamp())%14400 <= 14250:
                time.sleep(120)
            elif (datetime.utcnow().timestamp() - datetime(2024,1,1,0,0).timestamp())%14400 <= 14380:
                time.sleep(0.5)
            elif (datetime.utcnow().timestamp() - datetime(2024,1,1,0,0).timestamp())%14400 <= 14410:
                time.sleep(20)
                update_db(client)
                self.predict_current_4H()

                if self.test_mode == False:
                    self.is_active()
                    if self.active:
                        if (self.open_positions[0] == 0) and (self.current_4H_signal[1] - self.current_4H_signal[0] >= self.confs[self.turn]):
                            self.close_order(side='SELL')
                            self.open_positions = []
                            self.active = False
                            self.turn = 0
                        elif (self.open_positions[0] == 1) and (self.current_4H_signal[0] - self.current_4H_signal[1] >= self.confs[self.turn]):
                            self.close_order(side='BUY')
                            self.open_positions = []
                            self.active = False
                            self.turn = 0
                        else:
                            self.turn += 1
                    if self.active == False:
                        if self.current_4H_signal[0] - self.current_4H_signal[1] >= self.confs[self.turn]:
                            self.create_order(side='BUY')
                            self.open_positions.append(0)
                            self.turn += 1
                        elif self.current_4H_signal[1] - self.current_4H_signal[0] >= self.confs[self.turn]:
                            self.create_order(side='SELL')
                            self.open_positions.append(1)
                            self.turn += 1
                
                self.log_and_send_message()
                time.sleep(13000)
                client.ping()

bot = Bot()
bot.test_mode = True
bot.confidence_levels([0.5] + [0.17] + [0.15]*300)
bot.sl = 0.012
bot.tp = 0.018
bot.base_quantity = 0.025

bot.start()