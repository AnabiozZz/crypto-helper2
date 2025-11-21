import argparse
import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
from ch_indicators import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from ch2_ml import *
from ch2_validate import *
#plt.style.use('seaborn-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to ohlcv csv table', type=str, default="btcusd_1-min_data.csv")
parser.add_argument('-c', '--config', help='path to config file', type=str, default="testconfig.json")
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
candles_sliced['peak_cl'] = 0
peak_last = candles_sliced.iloc[0]
peak_new = candles_sliced.iloc[1]
diff_left_max = 0
for index, row in candles_sliced.iterrows():
    diff_left = (row[point_column] - peak_last[point_column])/peak_last[point_column]
    diff_right = (row[point_column] - peak_new[point_column])/peak_new[point_column]
    if (abs(diff_right) < abs(diff_left)) and (abs(diff_left) > abs(diff_left_max)):
        peak_new = row
        diff_left_max = diff_left
    elif (abs(diff_left_max) >= diff_threshold) and (abs(diff_right) >= diff_threshold):
        #print(f'!peak found: {index} = {row[point_column]}({diff_right}). apply {diff_left_max} to {peak_last.name}')
        candles_sliced.loc[peak_last.name, 'diff'] = diff_left_max
        candles_sliced.loc[peak_last.name, 'peak_cl'] = 1 if diff_left_max < 0 else -1
        peak_last = peak_new
        peak_new = row
        diff_left = diff_right
        diff_left_max = diff_left
        diff_right = 0

candles_sliced.to_csv('candles_sliced1.csv', index=True) 
candles_sliced.iloc[0,candles_sliced.columns.get_loc('diff')] = np.nan  #отсечение первого пика т.к. для него нет левого отклонения
candles_sliced['peak_high'] = candles_sliced['diff'] < 0
candles_sliced['peak_low'] = candles_sliced['diff'] > 0

### verify peaks
h_peaks_cnt = len(candles_sliced[candles_sliced.peak_high == True])
l_peaks_cnt = len(candles_sliced[candles_sliced.peak_low == True])
if (abs(h_peaks_cnt - l_peaks_cnt) > 1):
    print(f'FATAL ERROR: Peaks not balanced: {h_peaks_cnt}, {l_peaks_cnt}. Check peaks detection')
    exit(1)
## calculate indicators
indicators_conf = config["indicators"]
for i in indicators_conf:
    try:
        candles_sliced = indicators_f[i['type']](candles_sliced, i, ohlc_raw=candles)
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} {e}")
## Show plots
plt.style.use('seaborn-v0_8-whitegrid')
candles_sliced.to_csv('candles_sliced.csv', index=True) 
diff_view = candles_sliced.dropna()
diff_view.to_csv('diff_view.csv', index=True) 
peaks_view = diff_view.loc[:,"peak_high":]
ind_view_h = peaks_view.drop(['peak_high', 'peak_low'], axis=1)[peaks_view.peak_high == True]
ind_view_l = peaks_view.drop(['peak_high', 'peak_low'], axis=1)[peaks_view.peak_low == True]
fig = plt.figure(figsize=(25,13))
x_loc_fmt = mpl.dates.DateFormatter('%d.%m.%y')
n_rows = len(peaks_view.columns) + 1
n_cols = 3
plt_idx = 1
### prices
ax = fig.add_subplot(n_rows, n_cols, plt_idx)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.xaxis.set_major_formatter(x_loc_fmt)
ax.yaxis.set_major_locator(plt.MaxNLocator(8))
plt_idx += 1
ax.plot(diff_view.index, diff_view[point_column], color='red')
ax.text(0.5, 0.5, 'Price', fontsize=12, transform=ax.transAxes, ha='center', va='center', alpha=0.5)
### peaks
ax = fig.add_subplot(n_rows, n_cols, plt_idx)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.xaxis.set_major_formatter(x_loc_fmt)
ax.yaxis.set_major_locator(plt.MaxNLocator(8))
plt_idx += 1
ax.plot(diff_view.index, diff_view['diff'], 'o', color='red')
ax.text(0.5, 0.5, 'Difference', fontsize=12, transform=ax.transAxes, ha='center', va='center', alpha=0.5)
###stub
ax = fig.add_subplot(n_rows, n_cols, plt_idx) 
plt_idx += 1
for col in ind_view_h.columns:
    ### scatter plot
    ax = fig.add_subplot(n_rows, n_cols, plt_idx)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.xaxis.set_major_formatter(x_loc_fmt)
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    plt_idx += 1
    ax.text(0.5, 0.5, col, fontsize=12, transform=ax.transAxes, ha='center', va='center', alpha=0.5)
    ax.plot(ind_view_h.index, ind_view_h[col], 'v', color='red')
    ax.plot(ind_view_l.index, ind_view_l[col], '^', color='green')
    ### kde plot
    ax = fig.add_subplot(n_rows, n_cols, plt_idx)
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    plt_idx += 1
    sns.kdeplot(data=ind_view_h[col], fill=True, ax=ax, color='red')
    sns.kdeplot(data=ind_view_l[col], fill=True, ax=ax, color='green')
    ax.text(0.5, 0.5, col, fontsize=12, transform=ax.transAxes, ha='center', va='center', alpha=0.5)
    ### diff correlation plot
    ax = fig.add_subplot(n_rows, n_cols, plt_idx)
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    plt_idx += 1
    sns.jointplot(x=col, y="diff", data=diff_view, kind='reg', ax=ax)
    ax.text(0.5, 0.5, col, fontsize=12, transform=ax.transAxes, ha='center', va='center', alpha=0.5)
#fig.autofmt_xdate()
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
mplcursors.cursor(fig)
#plt.show()
##stage_regression
regression_conf = config["stage_regression"]
### correlation
corr_threshold = regression_conf["threshold"]
df_diff_ind = candles_sliced.loc[:, 'diff':].dropna()
df_diff_ind.drop(['peak_high','peak_low'], axis=1, inplace=True)
#print(df_diff_ind)
ind_to_drop = []
for ind in df_diff_ind.columns[:0:-1]:
    ind_corr = df_diff_ind.loc[:,['diff',ind]].corr(method='pearson').iloc[0,1]
    print(f'correlation for {ind}: {ind_corr}')
    if abs(ind_corr) < corr_threshold:
        ind_to_drop.append(ind)  
df_diff_ind.drop(ind_to_drop, axis=1, inplace=True)
if len(df_diff_ind.columns) <= 1:
    print('NO CORRELATIONS FOUND. EXIT')
    exit(1)

### linear regression
learn_test_ratio = regression_conf["learn_test_ratio"]
kfold = KFold(n_splits=learn_test_ratio, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, df_diff_ind.iloc[:, 1:], df_diff_ind['diff'], cv=kfold)
#print(results)
print("Linear regression MSE: mean = %.3f (stdev = %.3f)" % (results.mean(), results.std()))
### logistic regression
logist_ind = candles_sliced.loc[:,'peak_high':]
logist_ind.drop(ind_to_drop, axis=1, inplace=True)
logist_ind.dropna(inplace=True)
#print(logist_ind)
#### cross validation
kfold = StratifiedKFold(n_splits=learn_test_ratio, random_state=7, shuffle=True)
model = LogisticRegression(penalty=None)
results = cross_val_score(model, logist_ind.iloc[:,:1:-1], logist_ind['peak_high'], cv=kfold, scoring="roc_auc")
#print(results)
print("Logistic regression (high peak) MSE: mean = %.3f (stdev = %.3f)" % (results.mean(), results.std()))
model = LogisticRegression(penalty=None)
results = cross_val_score(model, logist_ind.iloc[:,:1:-1], logist_ind['peak_low'], cv=kfold, scoring="roc_auc")
#print(results)
print("Logistic regression (low peak) MSE: mean = %.3f (stdev = %.3f)" % (results.mean(), results.std()))
#### confusion matrix (high peak)
print(f"logist_ind high peaks:{len(logist_ind[logist_ind.peak_high == True])}")
X_train, X_test, Y_train, Y_test = train_test_split(logist_ind.iloc[:,:1:-1], logist_ind['peak_high'], 
                                                    test_size=1/learn_test_ratio, stratify=logist_ind['peak_high'])
model = LogisticRegression(penalty=None)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
print(f"Y_test high peaks:{len(Y_test[Y_test == True])}\nprediction high peaks:{len(prediction[prediction == True])} ")
print(matrix)

#### some uncommon test (high peak)
test_to_predict = Y_test.to_frame()
test_to_predict['prediction'] = prediction
test_to_predict = test_to_predict[test_to_predict.prediction == True]
print(test_to_predict)

print(f'Estimated profit:{check_profit()}')
### KDEClassifier
kde_data = candles_sliced.loc[:,'peak_cl':].dropna()
kde_x = kde_data.loc[:,'peak_low':]
kde_x.drop(['peak_low'], axis=1, inplace=True)
kde_y = kde_data['peak_cl']
X_train, X_test, Y_train, Y_test = train_test_split(kde_x, kde_y, 
                                                    test_size=1/learn_test_ratio, stratify=kde_y)
#### adjust bandwidth
grid = GridSearchCV(KDEClassifier(),
    {'bandwidth': np.logspace(0, 2, 100)})
grid.fit(X_train, Y_train)
fig, ax = plt.subplots()
ax.semilogx(np.array(grid.cv_results_['param_bandwidth']),
            grid.cv_results_['mean_test_score'])
ax.set(title='KDE Model Performance', ylim=(0, 1),
        xlabel='bandwidth', ylabel='accuracy')
#plt.show()
print(f'best param: {grid.best_params_}')
print(f'accuracy = {grid.best_score_}')
#### validation
model = KDEClassifier(bandwidth=grid.best_params_['bandwidth'])
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
print(f"""Y_test high peaks:{len(Y_test[Y_test == 1])} 
        prediction high peaks:{len(prediction[prediction == 1])}
        Y_test low peaks:{len(Y_test[Y_test == -1])} 
        prediction low peaks:{len(prediction[prediction == -1])}""")
print(matrix)
test_to_predict = Y_test.to_frame()
test_to_predict['prediction'] = prediction
print(test_to_predict)
print(f'failures: {test_to_predict[test_to_predict.peak_cl != test_to_predict.prediction]}')
    
        
