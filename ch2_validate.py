import pandas as pd

def check_profit(prices: pd.Series, prediction: pd.Series,
                stop_loss, take_profit):
    profit = 0
    td = prices.index[1] - prices.index[0]
    order_dir = 1 if take_profit > stop_loss else -1
    for pr_index, pr_row in prediction.items():
        price_order = prices.loc[pr_index]
        sl = price_order*stop_loss*order_dir
        tp = price_order*take_profit*order_dir
        for ohlc_index, ohlc_row in prices[pr_index+td:].items():
            diff = (ohlc_row - price_order)*order_dir
            if (diff >= tp):
                profit += abs(take_profit)
                break
            elif (diff <= sl):
                profit -= abs(stop_loss)
                break
    return profit