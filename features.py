import pandas as pd
import numpy as np
from datetime import datetime

def signal(x):
    #return x 
    if x > 1:
        return 0 #buy
    else:
        return 1 #sell
    
def signal3(x):
    #return x 
    if x > 1.003:
        return 0 #buy
    elif x < 0.9968:
        return 1 #sell
    else:
        return 2
    
def Revenue3_0(df, conf):
    dft = df.copy()
    dft['PredClass'] = dft['Preds'].apply(lambda x: np.argmax(x) if max(x) > (0.33 + conf) else 2)
    rev3 = dft.apply(lambda row: row['Close-Open'] if row['PredClass'] == 0 else -row['Close-Open'] if row['PredClass'] == 1 else 0, axis=1)
    return rev3

def Revenue3_1(df, conf):
    dft = df.copy()
    dft['PredClass'] = dft['Preds'].apply(lambda x: 0 if x[0] - x[1] > conf else 1 if x[1] - x[0] > conf else 2)
    rev3 = dft.apply(lambda row: row['Close-Open'] if row['PredClass'] == 0 else -row['Close-Open'] if row['PredClass'] == 1 else 0, axis=1)
    return rev3

def Moving_Average(df: pd.DataFrame, period:int) -> pd.Series:
    "Calculates moving averages for determined period"
    ma = df['Open'].rolling(period).mean()
    return ma

def Exp_Moving_Average(df: pd.DataFrame, period:int) -> pd.Series:
    "Calculates moving averages for determined period"
    ema = df['Open'].ewm(span=period, adjust=False).mean()
    return ema

def RSI(df: pd.DataFrame, period:int) -> pd.Series:
    "Calculates RSI for determined period"
    delta = df[f'Open'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_bollinger_bands(df, window=20, num_of_std=2):
    """
    Calculate Bollinger Bands for a given dataset.

    :param data: A DataFrame with a 'Close' column.
    :param window: The moving average window size (default is 20).
    :param num_of_std: The number of standard deviations away from the moving average (default is 2).
    :return: A DataFrame with Bollinger Bands columns added.
    """
    # Calculate the moving average (middle band)
    data = df.copy()
    data['Middle Band'] = data['Open'].rolling(window=window).mean()

    # Calculate the standard deviation
    std_dev = data['Open'].rolling(window=window).std()

    # Calculate the upper and lower bands
    data['Upper Band'] = data['Middle Band'] + (std_dev * num_of_std)
    data['Lower Band'] = data['Middle Band'] - (std_dev * num_of_std)

    return data[['Lower Band','Upper Band']]

def supertrend(df: pd.DataFrame, period:int, multiplier:int) -> pd.Series:
    "Calculates RSI for determined period"
    atr_multiplier = multiplier
    atr = df['High'] - df['Low']
    atr = atr.ewm(span=period, adjust=False).mean()
    upper_basic = (df['High'] + df['Low']) / 2 + (atr_multiplier * atr)
    lower_basic = (df['High'] + df['Low']) / 2 - (atr_multiplier * atr)
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for period in range(1, len(df)):
        if df['Close'][period - 1] > upper_band[period - 1]:
            upper_band[period] = min(upper_basic[period], upper_band[period - 1])
        else:
            upper_band[period] = upper_basic[period]

        if df['Close'][period - 1] < lower_band[period - 1]:
            lower_band[period] = max(lower_basic[period], lower_band[period - 1])
        else:
            lower_band[period] = lower_basic[period]

    dft = df.copy()
    dft['Trend'] = 1
    dft['Trend'][period] = 0

    for period in range(1, len(df)):
        if df['Close'][period] > upper_band[period - 1]:
            dft['Trend'][period] = 1
        elif df['Close'][period] < lower_band[period - 1]:
            dft['Trend'][period] = -1

    trend = dft['Trend']
    del dft
    return trend

def RSI_30MIN(df: pd.DataFrame, period:int) -> pd.Series:
    "Calculates RSI for determined period"
    rsi = df.apply(lambda row: list(pd.Series([row['Open']] + [row[f'30MIN_Open_T-{i}'] for i in range(1,period+1)]).diff()), axis=1)
    rsi.fillna(value=0, inplace=True)
    rsi = rsi.apply(lambda x: (sum([-t for t in x if t<=0])+1) / (sum([t for t in x if t>0])+1))
    rsi = rsi.apply(lambda x: 100 - (100/(1+x)))
    return rsi

def fed_interest(df):
    "returns fed interest rates"
    dft = df.copy()
    dft['fed'] = 0
    dft['fed'] = pd.to_datetime(df['Open_time']*pow(10,6)).apply(lambda x:
    1.5 if x > datetime(2017,12,14) else
    1.75 if x > datetime(2018,3,22) else
    2 if x > datetime(2018,6,14) else
    2.25 if x > datetime(2018,9,27) else
    2.5 if x > datetime(2018,12,20) else
    2.25 if x > datetime(2019,8,1) else
    2 if x > datetime(2019,9,19) else
    1.75 if x > datetime(2019,10,31) else
    0.75 if x > datetime(2020,3,17) else
    0.25 if x > datetime(2020,4,17) else
    0.5 if x > datetime(2022,5,5) else
    1 if x > datetime(2022,6,16) else
    1.75 if x > datetime(2022,7,27) else
    2.5 if x > datetime(2022,9,21) else
    3.25 if x > datetime(2022,11,2) else
    4.5 if x > datetime(2022,12,14) else
    4.75 if x > datetime(2023,2,1) else
    5 if x > datetime(2023,2,22) else
    5.5 if x > datetime(2023,7,26) else 1)
    return dft['fed']

def OBV(df: pd.DataFrame) -> pd.Series:
    "Calculates OBV"
    obv = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv[i] = obv[i - 1] + df['Volume'][i]
        elif df['Close'][i] < df['Close'][i - 1]:
            obv[i] = obv[i - 1] - df['Volume'][i]
        else:
            obv[i] = obv[i - 1]
    return obv

def AD(df: pd.DataFrame) -> pd.Series:
    "Calculates AD"
    ad = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) - (df['High'] - df['Low'])
    return ad

def AO(df: pd.DataFrame, period: int) -> pd.Series:
    "Calculates Aroon Oscillator"
    au = df['High'].rolling(window=period).apply(lambda x: x.argmax(), raw=True) / (period - 1) * 100
    ad = df['Low'].rolling(window=period).apply(lambda x: x.argmin(), raw=True) / (period - 1) * 100

    ao = au-ad
    return ao

def calculate_aroon_oscillator(df_tofunc, period=14):
    # Calculate Aroon-Up and Aroon-Down
    df_tofunc['Aroon-Up'] = df_tofunc['High'].rolling(window=period).apply(lambda x: x.argmax(), raw=True) / (period - 1) * 100
    df_tofunc['Aroon-Down'] = df_tofunc['Low'].rolling(window=period).apply(lambda x: x.argmin(), raw=True) / (period - 1) * 100
    
    # Calculate Aroon Oscillator
    df_tofunc['Aroon-Oscillator'] = df_tofunc['Aroon-Up'] - df_tofunc['Aroon-Down']
    
    return df_tofunc['Aroon-Oscillator']

def MACD(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9) -> pd.Series:
    "Calculates MA Convergence-Divergence"
    dft = df.copy()
    dft['ShortEMA'] = dft['Open'].ewm(span=short_window, adjust=False).mean()
    dft['LongEMA'] = dft['Open'].ewm(span=long_window, adjust=False).mean()
    dft['MACD_Ratio'] = dft['ShortEMA'] / dft['LongEMA']
    dft['MACD'] = dft['ShortEMA'] - dft['LongEMA']
    dft['SignalLine'] = dft['MACD'].ewm(span=signal_window, adjust=False).mean()
    dft['MACD-Histogram'] = dft['MACD'] - dft['SignalLine']
    return dft[['MACD_Ratio', 'SignalLine', 'MACD-Histogram']]
    
def Revenue(df: pd.DataFrame, conf: float) -> pd.Series:
    "Calculates revenue with given confidence threshold"
    rev = df.apply(lambda row: row['Close-Open'] if row['Preds'] > 0.5 + conf else -row['Close-Open'] if row['Preds'] < 0.5 - conf else 0, axis=1)
    return rev

def Calculate_ADX(df: pd.DataFrame, period: int) -> pd.Series:
    "Calculates Aroon Oscillator"
    df_tofunc = df.copy()
    df_tofunc['High-Low'] = df_tofunc['High'] - df_tofunc['Low']
    df_tofunc['High-PrevClose'] = abs(df_tofunc['High'] - df_tofunc['Close'].shift(1))
    df_tofunc['Low-PrevClose'] = abs(df_tofunc['Low'] - df_tofunc['Close'].shift(1))
    df_tofunc['TrueRange'] = df_tofunc[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # Calculate Directional Movement (DM)
    df_tofunc['High-Low-1'] = df_tofunc['High'].shift(1) - df_tofunc['Low'].shift(1)
    df_tofunc['High-PrevClose-1'] = abs(df_tofunc['High'].shift(1) - df_tofunc['Close'].shift(1))
    df_tofunc['Low-PrevClose-1'] = abs(df_tofunc['Low'].shift(1) - df_tofunc['Close'].shift(1))
    df_tofunc['DMplus'] = 0.0
    df_tofunc['DMminus'] = 0.0
    df_tofunc.loc[df_tofunc['High-Low'] > df_tofunc['High-Low-1'], 'DMplus'] = df_tofunc['High-Low']
    df_tofunc.loc[df_tofunc['High-Low'] < df_tofunc['High-Low-1'], 'DMminus'] = df_tofunc['High-Low-1']
    
    # Calculate Directional Index (DI)
    df_tofunc['DIplus'] = (df_tofunc['DMplus'].rolling(window=period).sum() / df_tofunc['TrueRange'].rolling(window=period).sum()) * 100
    df_tofunc['DIminus'] = (df_tofunc['DMminus'].rolling(window=period).sum() / df_tofunc['TrueRange'].rolling(window=period).sum()) * 100
    
    # Calculate DX (Directional Movement Index)
    df_tofunc['DX'] = abs(df_tofunc['DIplus'] - df_tofunc['DIminus']) / (df_tofunc['DIplus'] + df_tofunc['DIminus']) * 100
    
    # Calculate ADX (Average Directional Index)
    df_tofunc['ADX'] = df_tofunc['DX'].rolling(window=period).mean()
    
    return df_tofunc['ADX']

def ADX(data: pd.DataFrame, period: int):
    """
    Computes the ADX indicator.
    """
    
    df = data.copy()
    alpha = 1/period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    adx = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return adx

def calculate_stochastic_oscillator(df: pd.DataFrame, period: int, smoothing: int) -> pd.Series:
    # Calculate the %K line
    lowest_low = df['Low'].rolling(window=period).min()
    highest_high = df['High'].rolling(window=period).max()
    k_os = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Smooth the %K line with a moving average to calculate %D line
    d_os = k_os.rolling(window=smoothing).mean()
    
    return k_os, d_os

def qqe(df: pd.DataFrame, rsi_period: int, smoothing: int) -> pd.Series: 
    #QQE Factor
    rsi = RSI(df, rsi_period)
    sm_rsi = rsi.ewm(span=smoothing).mean()
    change = (sm_rsi - sm_rsi.shift(1)).abs()
    change = change.ewm(27).mean().ewm(27).mean()

    return change * 4.236
    
def TREND(df: pd.DataFrame, period: int) -> pd.Series:
    "Detect BULL or BEAR"
    fh = df['Open'].copy()
    sh = df['Open'].copy().shift(int(period/2))
    for p in range(1,int(period/2)):
        fh += df['Open'].shift(p)
    for p in range(int(period/2)+1,period):
        sh += df['Open'].shift(p)
    return fh/sh

def Count_30MIN_Signals(df: pd.DataFrame, period: int) -> pd.Series:
    "Counts 30Min signals for last 4 Hours"
    count = df['30MIN_OpenRatio_T-1'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    for i in range(2,period+1):
        count += df[f'30MIN_OpenRatio_T-{i}'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    return count

def Count_30MIN_Signals_ETH(df: pd.DataFrame, period: int) -> pd.Series:
    "Counts 30Min signals for last 4 Hours"
    count = df['30MIN_OpenRatio_T-1_ETH'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    for i in range(2,period+1):
        count += df[f'30MIN_OpenRatio_T-{i}_ETH'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    return count

def Count_30MIN_Signals_BNB(df: pd.DataFrame, period: int) -> pd.Series:
    "Counts 30Min signals for last 4 Hours"
    count = df['30MIN_OpenRatio_T-1_BNB'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    for i in range(2,period+1):
        count += df[f'30MIN_OpenRatio_T-{i}_BNB'].apply(lambda x: None if x == None else 1 if x > 1 else 0)
    return count

def First_Open_inDay(df: pd.DataFrame) -> pd.Series:
    "Returns Series containing the first 'Open' value for each date."
    dft = df.copy()
    dft['Date'] = pd.to_datetime(df['Open_time'] * 1000000).apply(lambda x: x.date())
    first_open_in_day = dft.groupby('Date',as_index=False)['Open'].first().rename(columns={'Open':'Day_Open'})
    dft = pd.merge(dft[['Date','Open']], first_open_in_day, on=['Date'], how='left')

    return dft['Day_Open']

# def First_Candle_inDay(df: pd.DataFrame) -> pd.Series:
#     "Returns Series containing the first 'Candle' value for each date."
#     dft = df.copy()
#     dft['Date'] = pd.to_datetime(df['Open_time'] * 1000000).apply(lambda x: x.date())
#     first_candle_in_day = dft.groupby('Date')['Candle'].first()

#     return first_candle_in_day

# def First_Open_inDay(df: pd.DataFrame) -> pd.Series:
#     "Returns Series containing the first 'HighClose_Ratio' value for each date."
#     dft = df.copy()
#     dft['Date'] = pd.to_datetime(df['Open_time'] * 1000000).apply(lambda x: x.date())
#     first_open_in_day = dft.groupby('Date')['Open'].first()

#     return first_open_in_day