def trendlines_auto(ohlc, conf, **kwargs):
    print(conf)
    return ohlc

def h_vol(ohlc, conf, **kwargs):
    print(conf)
    return ohlc

indicators_f = {'trendlines_auto': trendlines_auto,
                'h_vol': h_vol}
    
