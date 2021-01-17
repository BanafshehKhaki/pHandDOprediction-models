import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import re
import datetime
import sklearn.metrics as skm
import functions as func
import os
import warnings
warnings.filterwarnings("ignore")


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


def ARIMAregression(train, test, p, d, q):
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
        # print('predicted=%f, expected=%f' % (yhat, test[t]))
    # evaluate forecasts
    r2_score = skm.r2_score(test, predictions)
    print('Test r2_score: %.3f' % r2_score)
    # plot forecasts against actual outcomes
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()

    return predictions


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
    return predictions


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


def main():

    models = ['ARIMA']
    targets = ['dissolved_oxygen']  # , 'DOcategory', 'pHcategory']
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    n_job = -1
    # evaluate parameters
    p_values = range(1, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookTwo/output_Cat_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookTwo/output_Reg_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse',  'R2': 'R2'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            resultFileName = 'results_'+target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directory+resultFileName,
                            index=False, header=False)

            path = 'Sondes_data/train/train_data/'
            method = 'OrgData'

            for n_steps in [1]:
                for PrH_index in [1, 3, 6]:
                    files = [f for f in os.listdir(path) if f.endswith(
                        '.csv') and f.startswith(sondefilename)]
                    file = files[0]
                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    dataset = pd.read_csv(path+file)
                    ######################
                    # FOR AR and ARIMA
                    ######################
                    train = dataset[target]
                    custom_cv = custom_cv_2folds(
                        train, 1, PrH_index)

                    ######################
                    # FOR MA
                    ######################
                    # dataset = temporal_horizon(dataset, PrH_index, target)
                    # train = dataset[target]
                    # train_target = dataset['Target_'+target]
                    # custom_cv = func.custom_cv_2folds(train, 3)

                    ######################
                    # Cross Validation sets
                    ######################
                    i = 0
                    for train_index, test_index in custom_cv:
                        train_y = train[train_index].values
                        # train_y_targets = train_target[train_index].values #for MA
                        test_y = train[test_index].values
                        # test_y_targets = train_target[test_index].values  #for MA

                        # predictions = movingAverage(
                        #     train_y, train_y_targets, test_y, test_y_targets)

                        # predictions = AutoRegression(train_y, test_y)

                        # FOR ARIMA
                        for p in p_values:
                            for d in d_values:
                                for q in q_values:
                                    if p == q and d == q:
                                        print(p, d, q)
                                    else:
                                        print(p, d, q)
                                        predictions = ARIMAregression(
                                            train_y, test_y, p, d, q)

                                        if cat == 1:
                                            predictions = np.array(
                                                predictions).astype(int)

                                        fpath = 'predictions_' + method+target+'_Window' + \
                                            str(n_steps) + '_TH' + \
                                            str(PrH_index)+'_CV' + str(i) + \
                                            '_vals_'+str(p)+'_'+str(d) + \
                                            '_'+str(q)+'_'+file

                                        cm0 = func.forecast_accuracy(
                                            predictions, test_y, cat)

                                        plt.scatter(np.arange(len(test_y)),
                                                    test_y, s=1)
                                        plt.scatter(np.arange(len(predictions)),
                                                    predictions, s=1)
                                        plt.legend(['actual', 'predictions'],
                                                   loc='upper right')

                                        plt.savefig(directory+fpath+'.jpg')

                                        plt.close()

                                        data = {'Actual': test_y,
                                                'Predictions': predictions}

                                        df = pd.DataFrame(data=data)

                                        df.to_csv(directory+fpath, index=False)

                                        if cat == 1:
                                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                                    'file_names': fpath, 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': cm0[9]}
                                        elif cat == 0:
                                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                                    'file_names': fpath,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                                        df = pd.DataFrame(data=data, index=[0])
                                        df.to_csv(directory+resultFileName,
                                                  index=False, mode='a', header=False)

                        i = i+1


if __name__ == "__main__":
    main()
