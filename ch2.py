import argparse
import json
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to ohlcv csv table', type=str, default="btcusd_1-min_data.csv")
parser.add_argument('-c', '--config', help='path to config file', type=str, default="defconfig.json")
args = parser.parse_args()
pd.set_option('display.max_columns', None)

candles = pd.read_csv(args.input)
candles['datetime'] = pd.to_datetime(candles['Timestamp'], unit='s')
candles['volume_usdt'] = (candles['High'] + candles['Low']) / 2 * candles['Volume']
candles = candles.set_index('datetime')
with open(args.config, encoding="utf-8") as f:
    config = json.load(f)
## stage check
check_conf = config["stage_check"]
### check gaps
allow_gaps = check_conf["allow_gaps"]
period = candles.index[1] - candles.index[0]
dt_full_range = pd.date_range(start=candles.index.min(), end=candles.index.max(), freq=period)
missing_dates = dt_full_range.difference(candles.index)
if (len(missing_dates)):
    print(f'from {candles.index.min()} to {candles.index.max()} ,missing candles at:\n{missing_dates}')
### check values
price_min = check_conf["price_min"]
price_max = check_conf["price_max"]
volume_min = check_conf["volume_min"]
volume_max = check_conf["volume_max"]
illegal_data = candles[(candles.Open < price_min) |
                       (candles.Close < price_min) |
                       (candles.High < price_min) |
                       (candles.Low < price_min) |
                       (candles.Volume < volume_min) |
                       (candles.Open > price_max) |
                       (candles.Close > price_max) |
                       (candles.High > price_max) |
                       (candles.Low > price_max) |
                       (candles.volume_usdt > volume_max)]
if (len(illegal_data)):
    print(f'from {candles.index.min()} to {candles.index.max()} illegal numbers:\n{illegal_data}')
## stage prepare
prepare_conf = config["stage_prepare"]
### resample
resample_period = prepare_conf["resample_period"]
candles_resampled = candles.resample(resample_period).agg({'Open': 'first', 
                                                            'High': 'max',
                                                            'Low': 'min',
                                                            'Close': 'last',
                                                            'Timestamp': 'last',
                                                            'volume_usdt': 'sum',
                                                            'Volume': 'sum'})
### slice
slice_start = prepare_conf["slice_start"]
slice_end = prepare_conf["slice_end"]
candles_sliced = candles_resampled[slice_start:slice_end]
gaps = missing_dates[(missing_dates > candles_sliced.index[0]) &
                     (missing_dates < candles_sliced.index[-1])]
if (len(gaps)):
    print(f'WARNING!!! gaps in selected range:\n{gaps}')
    if not allow_gaps:
        print('exit')
        exit(1)
## stage peaks detect
peaks_conf = config["stage_peaks_detect"]
point_column = peaks_conf['point_column']
diff_threshold = peaks_conf['diff_threshold']
peak_last = candles_sliced.iloc[0]
peak_new = candles_sliced.iloc[0]
peak_new_idx = np.nan
diff_left_max = 0
diff_new = 0
for index, row in candles_sliced.iterrows():
    diff_left = (row[point_column] - peak_last[point_column])/peak_last[point_column]
    if (abs(diff_left) > abs(diff_left_max)):
        peak_new_idx = index
        peak_new = row
        diff_left_max = diff_left
    if (abs(diff_left_max) >= diff_threshold):
        diff_right = (row[point_column] - peak_new[point_column])/peak_new[point_column]
        if (abs(diff_right) >= diff_threshold):
            candles_sliced.loc[peak_new_idx, 'diff'] = diff_left_max
            peak_last = peak_new
            peak_new_idx = index
            peak_new = row
            diff_left = diff_right
            diff_left_max = diff_left
            diff_right = 0
            
print(candles_sliced.dropna())

        

    
        
