import pandas as pd

def check_profit(prices, prediction, stop_loss, take_profit):
    profit = 0
    td = pd.to_timedelta(candles_sliced['Timestamp'].iloc[0:2].diff(), unit='s').iloc[1]
    print(td)
    for pr_index, pr_row in test_to_predict.iterrows():
        price_buy = candles_sliced.loc[pr_index,point_column]
        for ohlc_index, ohlc_row in candles_sliced.loc[pr_index+td:].iterrows():
            diff = (ohlc_row[point_column] - price_buy)/price_buy
            if (abs(diff) > diff_threshold):
                profit += -diff
                break