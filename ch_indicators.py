import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

def trendlines_auto(ohlc, conf, **kwargs):
    print(conf)
    return ohlc

def h_vol(ohlc, conf, **kwargs):
    print(conf)
    return ohlc

def sma(ohlc: pd.DataFrame, conf, **kwargs):
    print(conf)
    column = conf["column"]
    n = conf["n"]
    for window_n in n:
        ohlc_sma = ohlc[column].rolling(window=window_n).mean()
        #print(ohlc_sma)
        ohlc[f'sma{window_n}'] = (ohlc[column] - ohlc_sma) / ohlc_sma
    print(ohlc.dropna())
    return ohlc

def ema(ohlc, conf, **kwargs):
    print(conf)
    column = conf["column"]
    n = conf["n"]
    #ohlc[f'rsi{window_n}'] = ohlc_rsi
    for w in n:
        ema = talib.EMA(ohlc[column], w)
        ohlc[f'ema{w}'] = (ohlc[column] - ema)/ema
    print(ohlc.dropna())
    return ohlc

def rsi(ohlc, conf, **kwargs):
    print(conf)
    n = conf["n"]
    #ohlc[f'rsi{window_n}'] = ohlc_rsi
    for w in n:
        rsi = talib.RSI(ohlc['Close'], w)
        ohlc[f'rsi{w}'] = rsi
    print(ohlc.dropna())
    return ohlc



indicators_f = {'trendlines_auto': trendlines_auto,
                'h_vol': h_vol,
                'ema': ema,
                'sma': sma,
                'rsi': rsi}
    
