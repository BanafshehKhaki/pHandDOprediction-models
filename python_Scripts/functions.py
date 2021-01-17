import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
from keras import backend as Kb
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from keras.utils import to_categorical
from keras.models import Sequential
# from keras.models import Functional
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.models import Model
from keras import metrics
import tensorflow as tf
from numpy import array
import seaborn as sns
from math import sqrt
from numpy import argmax
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.layers import Dropout
import keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier

param_grid = {'param_grid_DT1': {
    'model__class_weight': ['balanced'],  # {0: 1, 1: 50, 2: 100},
    'model__criterion': ['gini'],  # entropy
    'model__max_depth': np.arange(8, 15),
    'model__min_samples_split': np.arange(0.1, 1),
    'model__min_samples_leaf': np.arange(1, 6),
    'model__max_features': ['log2', 'auto', 'sqrt']},

    'param_grid_LR1': {
    'model__class_weight': [{0: 1, 1: 50, 2: 100}],
    "model__C": [0.001, 0.01],
    "model__penalty": ["l1", "l2"],
    'model__solver': ['saga']},

    'param_grid_LSTM1': {

    'model__batch_size': [64, 32, 128, 512],
    'model__epochs':  [20, 50, 100, 200],
    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
    'model__weight_constraint': [1, 2, 3, 4, 5],
    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
    'model__activation':  ['softmax', 'relu', 'tanh', 'sigmoid']},

    'param_grid_NN1': {

    'model__batch_size': [64, 32, 128, 512],
    'model__epochs':  [20, 50, 100, 200],
    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
    'model__weight_constraint': [1, 2, 3, 4, 5],
    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
    'model__activation':  ['softmax', 'relu', 'tanh', 'sigmoid']},

    'param_grid_SVC1': {
    'model__C': [10, 100],
    'model__class_weight': [{0: 1, 1: 50, 2: 100}, 'balanced'],
    'model__gamma': [1e-4, 0.01, 0.1, 'scale']},

    'param_grid_RF1': {
    'model__class_weight': [{0: 1, 1: 50, 2: 100}, 'balanced', 'balanced subsample'],
    # Maximum number of levels in tree
    'model__max_depth': [30, 50, 60, 100, None],
    # Number of trees in random forest
    'model__n_estimators': (30, 100),  # 1000
    # Number of features to consider at every split
    'model__max_features': ['auto', 'sqrt'],
    # Minimum number of samples required to split a node
    'model__min_samples_split': [5, 10],
    # Minimum number of samples required at each leaf node
    'model__min_samples_leaf': [1, 2, 4],
    # Method of selecting samples for training each tree
    'model__bootstrap': [True, False]},

    'param_grid_DT0': {
    'model__estimator__criterion': ['mae', 'mse', 'friedman_mse'],
    'model__estimator__max_depth': np.arange(8, 15),
    'model__estimator__min_samples_split': np.arange(0.1, 1),
    'model__estimator__min_samples_leaf': np.arange(1, 6),
    'model__estimator__max_features': ['log2', 'auto', 'sqrt'], },

    'param_grid_endecodeLSTM0': {
        'model__batch_size': [64, 32, 128, 512],
        'model__epochs':  [100, 200, 1000, 2000],
        'model__learn_rate': [0.0001, 0.001, 0.01, 0.1],
        'model__neurons': [32, 64, 128, 512],
        'model__activation':  ['relu', 'tanh', 'sigmoid']},

    'param_grid_ConvEnLSTM0': {
        'model__batch_size': [64, 32, 128, 512],
        'model__epochs':  [100, 200, 1000, 2000],
        'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
        'model__neurons': [32, 64, 128, 512],
        'model__activation':  ['relu', 'tanh', 'sigmoid']},

    'param_grid_CNNLSTM0': {
        'model__batch_size': [64, 32, 128, 512],
        'model__epochs':  [100, 200, 1000, 2000],
        'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
        'model__neurons': [32, 64, 128, 512],
        'model__activation':  ['relu', 'tanh', 'sigmoid']
},

    'param_grid_LR0': {
    # 'model__fit__estimator__alpha': [1e-3, 1e-2, 1e-1, 10, 100, 200, 300, 500, 600],
    'model__poly__estimator__degree': np.arange(10),
    # 'model__fit_estimator__intercept': [True, False],
    # 'model__fit__estimator__normalize': [True, False]
},

    'param_grid_LSTM0': {

    'model__batch_size': [64, 32, 128, 512],
    'model__epochs':  [100, 200, 1000, 2000],
    'model__learn_rate': [0.0001, 0.001, 0.01, 0.1],
    'model__weight_constraint': [1, 2, 3, 4, 5],
    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'model__neurons': [32, 64, 128, 512],
    'model__activation':  ['relu', 'tanh', 'sigmoid']},

    'param_grid_NN0': {

    'model__batch_size': [64, 32, 128, 512],
    'model__epochs':  [50, 100, 1000, 500],  # 500 added on 03.13.2020
    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
    'model__weight_constraint': [1, 2, 3, 4, 5],
    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'model__neurons': [24, 32, 64, 128, 512],
    'model__activation':  ['relu', 'tanh', 'sigmoid']},

    'param_grid_SVC0': {
    'model__estimator__C': [1.5, 10, 100],
    'model__estimator__gamma': [1e-4, 0.01, 0.1],
    'model__estimator__epsilon': [0.01, 0.1, 0.2, 0.5]},

    'param_grid_RF0': {
    # Maximum number of levels in tree
    'model__estimator__max_depth': [3, 30, 50, 60, 100, None],
    # Number of trees in random forest
    'model__estimator__n_estimators': (30, 100, 200, 400),  # 1000
    # Number of features to consider at every split
    'model__estimator__max_features': ['auto', 'sqrt'],
    # Minimum number of samples required to split a node
    'model__estimator__min_samples_split': [2, 5, 10],
    # Minimum number of samples required at each leaf node
    'model__estimator__min_samples_leaf': [1, 2, 4],
    # Method of selecting samples for training each tree
    'model__estimator__bootstrap': [True, False]}
    # estimator__ for MultiOutputRegressor() added
}

trained_param_grid = {
    'param_grid_CNNLSTM0': {
        'param_dissolved_oxygen_0': ['sigmoid', 512, 128, 200, 0.0001, 24],
        'param_ph_0': ['sigmoid', 512, 128, 200, 0.0001, 6],
    },
    'param_grid_LSTM0': {  # activation, neurons, batch_size, epochs, learn_rate, dropout_rate, weight_constraint, n_steps_in
        'param_dissolved_oxygen_0': ['tanh', 32, 512, 100, 0.01, 0.3, 4, 24],
        'param_ph_0': ['sigmoid', 512, 512, 200, 0.0001, 0.1, 1, 24],
    },
    'param_grid_LSTM_oneSonde0': {
        'param_dissolved_oxygen_0': ['tanh', 32, 512, 100, 0.001, 0.1, 5, 36],
        'param_ph_0': ['sigmoid', 32, 64, 100, 0.0001, 0.1, 3, 36],
    },
    'param_grid_ConvEnLSTM0': {
        'param_dissolved_oxygen_0': ['sigmoid', 512, 128, 200, 0.0001, 24],
        'param_ph_0': ['tanh', 64, 512, 200, 0.0001, 24],
    },
    'param_grid_endecodeLSTM0': {
        'param_dissolved_oxygen_0': ['tanh', 64, 512, 200, 0.0001, 12],
        'param_ph_0': ['tanh', 64, 512, 200, 0.0001, 24],
    },
    'param_grid_NN0': {  # activation, neuron, bach size, epoch, learning rate, drop rate, weight constraint, lag
        'param_dissolved_oxygen_0': ['tanh', 64, 64, 500, 0.0001, 0.5, 5, 24],
        'param_ph_0': ['tanh', 64, 64, 500, 0.0001, 0.5, 5, 24],
    },

    'param_grid_SVC0': {  # epsilon,gamma, C, lag
        'param_dissolved_oxygen_0': [0.5, 0.0001, 10, 24],
        'param_ph_0': [0.5, 0.0001, 1.5, 12],
    },

    'param_grid_RF_onereg0': {  # maxfeature, n_estimator, maxdepth, minsampleleaf, min_samplesplit, bootstrap, lag
        'param_dissolved_oxygen_0': ['sqrt', 100, 50, 2, 2, False, 12],
        'param_ph_0': ['auto', 400, 3, 2, 10, True, 6],
    },

    'param_grid_DT_onereg0': {  # split, maxfeatures, criterion, minsampleleaf, maxdepth, lag
        'param_dissolved_oxygen_0': [0.1, 'auto', 'mae', 5, 8, 12],
        'param_ph_0': [0.1, 'sqrt', 'mse', 4, 11, 6],
    },

    'param_grid_RF0': {  # maxfeature, n_estimator, maxdepth, minsampleleaf, min_samplesplit, bootstrap, lag
        'param_dissolved_oxygen_0': ['auto', 100, None, 4, 10, True, 24],
        'param_ph_0': ['auto', 400, 3, 2, 10, True, 3],
    },

    'param_grid_DT0': {  # split, maxfeatures, criterion, minsampleleaf, maxdepth, lag
        'param_dissolved_oxygen_0': [0.1, 'auto', 'mse', 3, 11, 6],
        'param_ph_0': [0.1, 'auto', 'mse', 3, 11, 3],
    },


}

#################################################
# Annual
#################################################
trained_param_grid_old = {
    'param_grid_RF1': {
        # max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap, n_steps, class_weights
        'param_DOcategory_1': [60, 30, 'sqrt', 10, 4, False, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_3': [60, 30, 'sqrt', 10, 4, False, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_6': [60, 30, 'auto', 10, 2, True, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_12': [60, 30, 'auto', 10, 2, True, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_24': [60, 30, 'sqrt', 10, 4, False, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_36': [50, 30, 'auto', 10, 2, False, 12, {0: 1, 1: 50, 2: 100}],
        'param_DOcategory_48': [60, 30, 'auto', 10, 2, True, 12, {0: 1, 1: 50, 2: 100}],
        # max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap, n_steps, class_weights
        'param_pHcategory_1': [100, 100, 'sqrt', 5, 4, True, 12, 'balanced'],
        'param_pHcategory_3': [100, 100, 'sqrt', 5, 4, True, 6, 'balanced'],
        'param_pHcategory_6': [100, 100, 'sqrt', 5, 4, True, 1, 'balanced'],
        'param_pHcategory_12': [100, 100, 'sqrt', 5, 4, True, 12, 'balanced'],
        'param_pHcategory_24': [100, 30, 'auto', 10, 2, True, 6, 'balanced'],
        'param_pHcategory_36': [100, 100, 'sqrt', 5, 4, True, 12, 'balanced'],
        'param_pHcategory_48': [100, 100, 'sqrt', 5, 4, True, 12, 'balanced'],
    },
    'param_grid_RF0': {
        # max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap, n_steps,
        'param_dissolved_oxygen_1': [100, 400, 'auto', 10, 1, True, 12],
        'param_dissolved_oxygen_3': [100, 200, 'auto', 10, 1, True, 6],
        'param_dissolved_oxygen_6': [100, 200, 'auto', 10, 1, True, 6],
        'param_dissolved_oxygen_12': [50, 100, 'sqrt', 2, 2, False, 12],
        'param_dissolved_oxygen_24': [100, 400, 'auto', 10, 1, True, 12],
        'param_dissolved_oxygen_36': [100, 400, 'auto', 10, 1, True, 12],
        'param_dissolved_oxygen_48': [100, 100, 'auto', 10, 2, True, 3],

        'param_ph_1': [],
        'param_ph_3': [],
        'param_ph_6': [],
        'param_ph_12': [],
        'param_ph_24': [],
        'param_ph_36': [],
        'param_ph_48': [],
    },
    'param_grid_SVC1': {

        'param_DOcategory_1': [{0: 1, 1: 50, 2: 100}, 0.0001, 100, 6],
        'param_DOcategory_3': [{0: 1, 1: 50, 2: 100}, 0.0001, 10, 12],
        'param_DOcategory_6': [{0: 1, 1: 50, 2: 100},  0.0001, 10, 12],
        'param_DOcategory_12': ['balanced', 0.0001, 100, 12],
        'param_DOcategory_24': ['balanced', 0.0001, 10, 12],
        'param_DOcategory_36': ['balanced', 0.0001,  100, 12],
        'param_DOcategory_48': ['balanced', 0.0001, 100, 6],

        'param_pHcategory_1': ['balanced', 0.0001, 10, 1],
        'param_pHcategory_3': [{0: 1, 1: 50, 2: 100}, 0.01, 10, 3],
        'param_pHcategory_6': [{0: 1, 1: 50, 2: 100}, 0.0001, 100, 12],
        'param_pHcategory_12': [{0: 1, 1: 50, 2: 100}, 0.01, 10, 3],
        'param_pHcategory_24': [{0: 1, 1: 50, 2: 100}, 0.0001, 100, 12],
        'param_pHcategory_36': [{0: 1, 1: 50, 2: 100}, 0.01, 10, 1],
        'param_pHcategory_48': [{0: 1, 1: 50, 2: 100},  0.0001, 100, 12],
    },
    'param_grid_SVC0': {

        'param_dissolved_oxygen_1': [],
        'param_dissolved_oxygen_3': [],
        'param_dissolved_oxygen_6': [],
        'param_dissolved_oxygen_12': [],
        'param_dissolved_oxygen_24': [],
        'param_dissolved_oxygen_36': [],
        'param_dissolved_oxygen_48': [],

        # epsilon, gamma, C, n_steps
        'param_ph_1': [0.01, 0.0001, 10, 12],
        'param_ph_3': [0.01, 0.0001, 1.5, 12],
        'param_ph_6': [0.01, 0.0001, 1.5, 12],
        'param_ph_12': [0.01, 0.0001, 1.5, 12],
        'param_ph_24': [0.01, 0.01, 1.5, 1],
        'param_ph_36': [0.1, 0.01, 1.5, 1],
        'param_ph_48': [0.1, 0.01, 1.5, 3],
    },

    'param_grid_LR0': {

        'param_dissolved_oxygen_1': [],
        'param_dissolved_oxygen_3': [],
        'param_dissolved_oxygen_6': [],
        'param_dissolved_oxygen_12': [],
        'param_dissolved_oxygen_24': [],
        'param_dissolved_oxygen_36': [],
        'param_dissolved_oxygen_48': [],
        # model__fit__alpha, n_steps
        'param_ph_1': [650, 12],
        'param_ph_3': [650, 12],
        'param_ph_6': [550, 12],
        'param_ph_12': [650, 12],
        'param_ph_24': [550, 12],
        'param_ph_36': [650, 12],
        'param_ph_48': [650, 12],
    },
    'param_grid_NN1': {
        'param_DOcategory_1': [],
        'param_DOcategory_3': [],
        'param_DOcategory_6': [],
        'param_DOcategory_12': [],
        'param_DOcategory_24': [],
        'param_DOcategory_36': [],
        'param_DOcategory_48': [],

        'param_pHcategory_1': [2,  200,  'sigmoid', 0.0,  64,  32,  0.001, 12],
        'param_pHcategory_3': [2, 200, 'sigmoid',  0.0,  64,  32,  0.001, 12],
        'param_pHcategory_6': [4, 100,  'relu', 0.4, 32,  64, 0.01, 1],
        'param_pHcategory_12': [2,  200, 'sigmoid', 0.0, 64, 32, 0.001, 12],
        'param_pHcategory_24': [2, 200,  'sigmoid', 0.0, 64,  32,  0.001, 3],
        'param_pHcategory_36': [2, 200,  'sigmoid',  0.0,  64, 32, 0.001, 6],
        'param_pHcategory_48': [5,  50, 'softmax', 0.1, 64,  64, 0.01, 1],
    },

    'param_grid_dummy1': {
        'param_DOcategory_1': [],
        'param_DOcategory_3': [],
        'param_DOcategory_6': [],
        'param_DOcategory_12': [],
        'param_DOcategory_24': [],
        'param_DOcategory_36': [],
        'param_DOcategory_48': [],

        'param_pHcategory_1': [],
        'param_pHcategory_3': [],
        'param_pHcategory_6': [],
        'param_pHcategory_12': [],
        'param_pHcategory_24': [],
        'param_pHcategory_36': [],
        'param_pHcategory_48': [],
    },
    'param_grid_dummy0': {
        'param_dissolved_oxygen_1': [],
        'param_dissolved_oxygen_3': [],
        'param_dissolved_oxygen_6': [],
        'param_dissolved_oxygen_12': [],
        'param_dissolved_oxygen_24': [],
        'param_dissolved_oxygen_36': [],
        'param_dissolved_oxygen_48': [],

        'param_ph_1': [],
        'param_ph_3': [],
        'param_ph_6': [],
        'param_ph_12': [],
        'param_ph_24': [],
        'param_ph_36': [],
        'param_ph_48': [],
    },
}


def algofind(modelname, input_dim, n_steps, cat):
    if cat == 1:
        if modelname == 'LSTM':
            model = KerasClassifier(build_fn=create_LSTM_model, input_dim=input_dim,
                                    epochs=20, batch_size=64,  nsteps=int(n_steps), verbose=0)
        elif modelname == 'DT':
            model = DecisionTreeClassifier()  # OneVsRestClassifier(
        elif modelname == 'RF':
            model = RandomForestClassifier()
        elif modelname == 'LR':
            model = LogisticRegression(
                multi_class='multinomial', max_iter=2000)
        elif modelname == 'SVC':
            model = SVC()
        elif modelname == 'NN':
            model = KerasClassifier(build_fn=create_NN_model, epochs=20, batch_size=64,
                                    input_dim=input_dim, verbose=0)

    elif cat == 0:
        if modelname == 'LSTM':
            model = KerasRegressor(build_fn=create_reg_LSTM_model, input_dim=input_dim,
                                   epochs=20, batch_size=64,  nsteps=int(n_steps), verbose=0)
        elif modelname == 'DT':
            model = DecisionTreeRegressor()
        elif modelname == 'RF':
            model = RandomForestRegressor()
        elif modelname == 'LR':
            model = Pipeline(
                [('poly', PolynomialFeatures()), ('fit', Ridge())])
        elif modelname == 'SVC':
            model = SVR()
        elif modelname == 'NN':
            model = KerasRegressor(build_fn=create_reg_NN_model, epochs=20, batch_size=64,
                                   input_dim=input_dim, verbose=0)

    return model


# def custom_score(test_y, predictions):
#     test_y = np.argmax(test_y, axis=-1)
#     predictions = np.argmax(predictions, axis=-1)
#     F1_01 = skm.f1_score(test_y, predictions, labels=[1, 2], average='micro')
#     return F1_01


def forecast_accuracy(predictions, test_y, cat):
    if cat == 1:
        F1 = skm.f1_score(test_y, predictions, labels=[
                          1, 2], average=None).ravel()

        P = skm.precision_score(test_y, predictions, labels=[
                                1, 2], average=None).ravel()

        R = skm.recall_score(test_y, predictions, labels=[
            1, 2], average=None).ravel()

        # tp, fn, fp, tn = confusion_matrix(
        #     test_y, predictions, labels=[1, 2]).ravel()
        # print(tp, fn, fp, tn)
        # acc = (tp+tn)/(tp+fp+fn+tn)
        acc = 0

        F1_0_1 = skm.f1_score(test_y, predictions, labels=[
                              1, 2], average='micro')

        F1_all = skm.f1_score(test_y, predictions, average='micro')

        fbeta = skm.fbeta_score(test_y, predictions, labels=[
                                1, 2],  beta=2, average='micro')

        return(F1[0], F1[1], P[0], P[1], R[0], R[1], acc, F1_0_1, F1_all, fbeta)
    else:
        test_y = test_y + 0.0000001
        mape = np.mean(np.abs(predictions - test_y)/np.abs(test_y))  # MAPE
        me = np.mean(predictions - test_y)             # ME

        # np.mean((predictions - test_y)/test_y)   # MPE

        # mae = np.mean(np.abs(predictions - test_y))    # MAE
        mae = skm.mean_absolute_error(test_y, predictions)

        # rmse = np.sqrt(np.mean((predictions - test_y)**2))  # RMSE
        mse = skm.mean_squared_error(test_y, predictions)
        rmse = np.sqrt(skm.mean_squared_error(test_y, predictions))

        # corr = np.corrcoef(predictions, test_y)[0, 1]   # corr

        r2 = skm.r2_score(test_y, predictions)       # R2
        # 1-(sum((predictions - test_y)**2)/sum((test_y-np.mean(test_y))**2))

        return(mape, me, mae, mse, rmse, r2)


def inverseTransform(predictions, test_y, method, file, path):
    y_scaler_path = path
    y_scaler_filename = re.sub('.csv', '_'+method+'_y.save', file)
    y_scaler = joblib.load(y_scaler_path+y_scaler_filename)
    inv_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    inv_yhat = y_scaler.inverse_transform(predictions.reshape(-1, 1))
    return inv_yhat, inv_y


def transform(predictions, test_y, method, target, file):
    # print(predictions.shape)
    path = 'Sondes_data/train/train_data_normalized/' + \
        method+'/'+target+'/'
    if method == 'MinMaxScaler' or method == 'StandardScaler':
        inv_yhat, inv_y = inverseTransform(
            predictions, test_y, method, file, path)
    else:
        inv_yhat, inv_y = predictions, test_y
    return inv_yhat, inv_y


def split_sequences(data, n_steps):
    data = data.values
    X, y = list(), list()

    for i in range(len(data)):
        end_ix = i + n_steps*6
        if end_ix > len(data):
            break

        Kx = np.empty((1, 12))
        for index in np.arange(i, i+(n_steps*6), step=6, dtype=int):
            eachhour = index + 6
            if eachhour > len(data) or i+(n_steps*6) > len(data):
                break

            a = data[index: eachhour, : -1]
            hourlymean_x = np.round(np.mean(a, axis=0), decimals=2)
            hourlymean_y = data[eachhour-1, -1]

            hourlymean_x = hourlymean_x.reshape((1, hourlymean_x.shape[0]))
            if index != i:
                Kx = np.append(Kx, hourlymean_x, axis=0)
            else:
                Kx = hourlymean_x

        X.append(Kx)
        y.append(hourlymean_y)
    # print(np.array(X).shape)
    return np.array(X), np.array(y)


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

######################################
# Creating custom datasets
# By choosing a random minutes from each hour to represent that hour
######################################


def custom_cv_2folds(X, kfolds):
    n = X.shape[0]
    # print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-6, step=6, dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        # print(idx[0:10])
        yield idx[:int(len(idx)*0.7)], idx[int(len(idx)*0.7):]
        i = i+1


def custom_cv_kfolds_testdataonly(X, kfolds):
    n = X.shape[0]
    # print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-6, step=6, dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        # print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1


def create_LSTM_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='relu', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Reshape(target_shape=(
        nsteps, input_dim[2]), input_shape=(nsteps*input_dim[2],)))
    model.add(LSTM(neurons, activation=activation, return_sequences=True,
                   kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, activation=activation, return_sequences=True))
    model.add(LSTM(neurons, activation=activation))  # Adding new layer
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['acc', keras.losses.categorical_crossentropy])
    print('model: ' + str(model))
    return model


def create_reg_LSTM_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Reshape(target_shape=(
        nsteps, input_dim[2]), input_shape=(nsteps*input_dim[2],)))
    model.add(LSTM(neurons, activation=activation, return_sequences=True,
                   kernel_constraint=maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, activation=activation, return_sequences=True))
    model.add(LSTM(neurons, activation=activation))  # Adding new layer
    model.add(Dense(1))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mae', optimizer=opt)
    print('model: ' + str(model))
    return model


def create_reg_NN_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Dense(neurons, activation=activation,
                    kernel_constraint=maxnorm(weight_constraint), input_shape=(input_dim[1]*input_dim[2],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))  # adding new layer
    model.add(Dense(1))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mae', optimizer=opt)
    print('model: ' + str(model))
    return model


def setfeatures(currentlist, n_steps):
    # Creating name of the new columns of data:
    columns = list(currentlist)
    for i in range(1, n_steps):
        w = '+'+str(i)+'h'
        a = [w + str(item) for item in currentlist]
        columns.extend(a)
    # print(columns)
    return columns


def getlags_window(model_name, params, cat):
    if cat == 1:
        if model_name == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps, class_weight = params
        else:
            n_steps = params[-1]
    if cat == 0:
        n_steps = params[-1]
    return n_steps


def create_NN_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='relu', input_dim=None):
    model = Sequential()
    model.add(Dense(neurons, activation=activation,
                    kernel_constraint=maxnorm(weight_constraint), input_shape=(input_dim[1]*input_dim[2],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))  # adding new layer
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[
        'acc', keras.losses.categorical_crossentropy])
    print('model: ' + str(model))
    return model


# input_dim is for LSTM to create the 3D shape
def getModel(model, input_dim, params, n_jobs, cat):
    if cat == 1:
        if model == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps, class_weight = params
            clf = RandomForestClassifier(max_depth=max_depth,
                                         bootstrap=bootstrap,
                                         n_estimators=n_estimators,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features,
                                         class_weight=class_weight, n_jobs=n_jobs)
        if model == 'SVC':
            class_weight, gamma, C, n_steps = params
            clf = SVC(class_weight=class_weight,
                      gamma=gamma, C=C)
        if model == 'NN':
            model_weight_constraint, model_epochs, model_activation, model_dropout_rate, model_neurons, model_batch_size, model_learn_rate, n_steps = params
            clf = KerasClassifier(build_fn=create_NN_model, weight_constraint=model_weight_constraint, epochs=model_epochs, activation=model_activation,
                                  dropout_rate=model_dropout_rate, neurons=model_neurons, batch_size=model_batch_size, input_dim=input_dim, learn_rate=model_learn_rate, verbose=0)

    else:
        if model == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps = params
            clf = RandomForestRegressor(max_depth=max_depth,
                                        bootstrap=bootstrap,
                                        n_estimators=n_estimators,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, n_jobs=n_jobs)
        if model == 'SVC':
            epsilon, gamma, C, n_steps = params
            clf = SVR(epsilon=epsilon, gamma=gamma, C=C)
        if model == 'LR':
            alpha, n_steps = params
            clf = Pipeline(
                [('poly', PolynomialFeatures()), ('fit', Ridge(alpha=alpha))])

    return clf


def preparedata(dataset, PrH_index, lags, target, cat):
    # dataset = dataset.drop(
    #     columns=['time', 'depth', 'lat', 'lon', 'year'])

    dataset = dataset.dropna()
    print(dataset.head())

    # horizontally stack columns
    # dataset = hstack((dataset1, dataset2, dataset3))

    dataset = temporal_horizon(
        dataset, PrH_index, target)

    train_X_grid, train_y_grid = split_sequences(
        dataset, lags)

    input_dim = train_X_grid.shape

    # print('na:')
    inds = np.where(np.isnan(train_X_grid))
    # print(inds)
    train_X_grid[inds] = 0
    inds = np.where(np.isnan(train_y_grid))
    train_y_grid[inds] = 0
    # print(inds)
    # print('--')

    train_X_grid = train_X_grid.reshape(
        train_X_grid.shape[0], train_X_grid.shape[1]*train_X_grid.shape[2])

    # get a warning if 1 is out in as second parameter: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    train_y_grid = train_y_grid.reshape(len(train_y_grid),)

    if cat == 1:
        train_y_grid = train_y_grid.astype(int)

    return train_X_grid, train_y_grid, input_dim, list(dataset.columns[:-1])
