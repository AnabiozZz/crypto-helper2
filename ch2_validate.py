import pandas as pd
from enum import IntEnum
import backtrader as bt
import datetime

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

def run_backtrader(config, ohlcv: pd.DataFrame, predictions: pd.Series):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=ohlcv, openinterest=None)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
    cerebro.addstrategy(MLStrategy, predictions=predictions,
                        bet=config['bet'],
                        take_profit=config['take_profit'],
                        stop_loss=config['stop_loss'],
                        close_opposite=config['close_opposite'])
    cerebro.run()


class MLStrategy(bt.Strategy):
    params = (
        ('predictions', None),
        ('bet', 1),
        ('take_profit', 0.04),
        ('stop_loss', 0.04),
        ('close_opposite', True)
        
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED', dt=order.created.dt)
            self.order = order
            return

        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')

        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

        else:
            self.log(f'UNKNOWN ORDER STATUS: {order.status}')
        # Sentinel to None: new orders allowed
        self.order = None

    def __init__(self):
        # Sentinel to None: new ordersa allowed
        self.order = None
        self.prediction_dt = self.p.predictions.index[0]

    def next(self):
        if self.order:
            # An order is pending ... nothing can be done
            return

        if (self.data.datetime.datetime(0) != self.prediction_dt):
            return
        prediction_val = self.p.predictions[self.prediction_dt]
        next_prediction_idx = self.p.predictions.index.get_loc(self.prediction_dt) + 1
        if next_prediction_idx == len(self.p.predictions):
            return
        self.prediction_dt = self.p.predictions.index[next_prediction_idx]
        self.log(f'Prediction: {prediction_val}, estimated value:{self.broker._value}')
        if (prediction_val == 'None'):
            return
        # Check if we are in the market
        if self.position:
            # In the maerket - check if it's the time to sell
            if prediction_val == 'High':
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.sell()

        elif prediction_val == "Low":
                self.buy(exectype=bt.Order.Market)  # default if not given

                self.log('BUY CREATE, exectype Market, price %.2f' %
                         self.data.close[0])
