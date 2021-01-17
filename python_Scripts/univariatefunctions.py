import functions as func
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
import sys
import time
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import re
import datetime
import sklearn.metrics as skm
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os


def custom_cv_kfolds_testdataonly(X, kfolds, th):
    n = X.shape[0]
    # print(n)
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1


def temporal_horizon(df, pd_steps, target):
    pd_steps = pd_steps * 6
    target_values = df[[target]]
    target_values = target_values.drop(
        target_values.index[0: pd_steps], axis=0)
    target_values.index = np.arange(0, len(target_values[target]))

    df = df.drop(
        df.index[len(df.index)-pd_steps: len(df.index)], axis=0)
    df['Target_'+target] = target_values
    print('Target_'+target)
    return df


def ARIMAregression(train, test, config,  cat, directory, file, target, PrH_index, n_steps, CV, result_filename, timeline):
    p, d, q = config
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(trend='nc', disp=0)

        # plot some history and the forecast with confidence intervals
        # model_fit.plot_predict(len(train)-10, len(train)+1)
        # plt.legend(loc='upper left')
        # plt.show()

        output, stderr, conf = model_fit.forecast()

        # summarize forecast and confidence intervals
        # print('Expected: %.3f' % test[t])
        # print('Forecast: %.3f' % output)
        # print('Standard Error: %.3f' % stderr)
        # print('95%% Confidence Interval: %.3f to %.3f' % (conf[0][0], conf[0][1]))

        yhat = output
        predictions.append(yhat)
        history.append(test[t])
        # print(predictions[-1])
        # print('predicted=%f, expected=%f' % (yhat, test[t]))
    # evaluate forecasts
    r2_score = skm.r2_score(test, predictions)
    print('Test r2_score: %.3f' % r2_score)
    # plot forecasts against actual outcomes
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()
    saveResults(predictions, test,  cat, directory, file,
                target, PrH_index, n_steps, CV, result_filename, timeline, config)
    return r2_score


def movingAverage(train_X, train_y, test_X, test_y):

    # persistence model on training set
    train_pred = [x for x in train_X]
    # calculate residuals
    train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]

    # model the training set residuals
    model = AR(train_resid)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train_resid[len(train_resid)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    # expected_error = list()
    for t in range(len(test_y)):
        # persistence
        yhat = test_X[t]
        error = test_y[t] - yhat
        # expected_error.append(error)
        # predict error
        length = len(history)
        lag = [history[i] for i in range(length-window, length)]
        pred_error = coef[0]
        for d in range(window):
            pred_error += coef[d+1] * lag[window-d-1]
        yhat = yhat + pred_error
        predictions.append(yhat)
        history.append(error)
        # print('predicted error=%f, expected error=%f' %
        #       (pred_error, error))
    # plt.plot(test_y)
    # plt.plot(predictions, color='red')
    # plt.legend()
    # plt.show()
    return predictions


def AutoRegression(train, test):

    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    # print(len(history))
    history = [history[i]for i in range(len(history))]
    # print(history[0:5])
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window, length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)  # new observations added to history
        # print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions, window, coef


def custom_cv_2folds(X, kfolds, th):
    n = X.shape[0]
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0: 10])
        yield idx[: int(len(idx)*0.7)], idx[int(len(idx)*0.7):]
        i = i+1


# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast_onestep(history, config):
    t, d, s, p, b, r = config
    # define model
    history = array(history)
    model = ExponentialSmoothing(
        history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

####
# walk-forward validation for exp_smoothing_forecast


def exp_smoothing_forecast(train, test, cfg, cat, directory, file, target, PrH_index, n_steps, CV, result_filename, timeline):
    predictions = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = exp_smoothing_forecast_onestep(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # print(predictions[-1])
    # estimate prediction error
    saveResults(predictions, test,  cat, directory, file,
                target, PrH_index, n_steps, CV, result_filename, timeline, cfg)
    r2_score = skm.r2_score(test, predictions)
    print(r2_score)
    return r2_score

# one-step sarima forecast


def sarima_forecast_oneStep(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False, maxiter=2000)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    # print(yhat[0])
    return yhat[0]


def sarima_forecast(train, test, cfg,  cat, directory, file, target, PrH_index, n_steps, CV, result_filename, timeline):
    predictions = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast_oneStep(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # print(predictions[-1])
    saveResults(predictions, test,  cat, directory, file,
                target, PrH_index, n_steps, CV, result_filename, timeline, cfg)

    r2_score = skm.r2_score(test, predictions)
    return r2_score


def exp_smoothing_configs():
    models = list()
    # define config lists
    t_params = ['add', 'mul']  # None
    d_params = [True, False]
    s_params = [None]  # 'add', 'mul']  # None
    p_params = [None, 24*30, 24*30*5]
    b_params = [False]  # True
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
    return models


def score_model(model, train, test,  cfg,   cat, directory, file, target, PrH_index, n_steps, CV, result_filename, timeline, debug=False):
    print(model)
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = model(train, test, cfg,  cat, directory, file,
                       target, PrH_index, n_steps, CV, result_filename, timeline)
    else:
        # one failure during model validation suggests an unstable config
        try:
                        # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                if model == 'SARIMA':
                    result = sarima_forecast(train, test, cfg,  cat, directory, file,
                                             target, PrH_index, n_steps, CV, result_filename, timeline)
                elif model == 'ARIMA':
                    result = ARIMAregression(train, test, cfg,  cat, directory, file,
                                             target, PrH_index, n_steps, CV, result_filename, timeline)
                elif model == 'ETS':
                    result = exp_smoothing_forecast(train, test, cfg,  cat, directory, file,
                                                    target, PrH_index, n_steps, CV, result_filename, timeline)
        except:
            error = None
    # check for an interesting result
        if result is not None:
            print(' > Model[%s] %.3f' % (key, result))
        return (key, result)


def ARIMA_configs():
    models = list()
    # define config lists
    p_values = range(1, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    # create config instances
    for p in p_values:
        for d in d_values:
            for q in q_values:
                cfg = [p, d, q]
                models.append(cfg)
    return models


# create a set of sarima configs to try
# order: A tuple p, d, and q parameters for the modeling of the trend.
# seasonal order: A tuple of P, D, Q, and m parameters for the modeling the seasonality
# trend: A parameter for controlling a model of the deterministic trend as one of ‘n’, ‘c’, ‘t’, and ‘ct’ for no trend, constant, linear, and constant with linear trend, respectively.

def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


def saveResults(predictions, test_y, cat, directory, file, target, PrH_index, n_steps, i, result_filename, timeline, config):
    print(cat, directory, file, target,
          PrH_index, n_steps, i, result_filename, config)
    if cat == 1:
        predictions = np.array(predictions).astype(int)
        test_y = np.array(test_y).astype(int)

    cm0 = func.forecast_accuracy(
        predictions, test_y, cat)

    filename = file + '_' + \
        target+'_TH' + \
        str(PrH_index)+'_lag' + \
        str(n_steps)+'_'+str(i)+'_config'+str(config)

    directorydeeper = directory+'more/'
    if not os.path.exists(directorydeeper):
        os.makedirs(directorydeeper)

    data = {'time': timeline,
            'Actual': test_y,
            'Predictions': predictions}
    df = pd.DataFrame(data=data)

    df.to_csv(directorydeeper+filename +
              '.csv', index=False)

    plt.scatter(timeline.values,
                test_y, s=1)
    plt.scatter(timeline.values,
                predictions, s=1)
    plt.legend(['actual', 'predictions'],
               loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig(directorydeeper+filename+'.png')
    plt.close()

    # print(directorydeeper)
    # print(filename)
    # print(cm0)

    method = 'OrgData'
    if cat == 1:
        data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1, 'config': [config],
                'file_names': file,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
    elif cat == 0:
        data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1, 'config': [config],
                'file_names': file,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

    df = pd.DataFrame(data=data, index=[0])
    df.to_csv(directory+result_filename,
              index=False, mode='a', header=False)
    print(directory+result_filename)
    print('-------------------------')
