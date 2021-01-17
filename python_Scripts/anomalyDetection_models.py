from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import os
import re
import time
import joblib
import gc
from keras import backend as Kb
import numpy as np
import pandas as pd
from sklearn import tree
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from keras.models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import functions as func
from keras.utils import to_categorical
from numpy import argmax
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from numpy import vstack


def R2_measure(y_true, y_pred):
    return r2_score(y_true, y_pred)


def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, labels=[1, 2],  beta=2, average='micro')

# define models to test


def get_models():
    models, names = list(), list()
    # LOF
    # models.append(LocalOutlierFactor(contamination=0.01))
    # names.append('LOF')
    # # EE
    # models.append(EllipticEnvelope(contamination=0.01))
    # names.append('EE')
    # IF
    models.append(IsolationForest(contamination=0.01))
    names.append('IF')
    # SVM
    models.append(OneClassSVM(gamma='scale', nu=0.01))
    names.append('SVM')
    return models, names

# make a prediction with a lof model


def lof_predict(model, trainX, testX):
    # create one large dataset
    composite = vstack((trainX, testX))
    # make prediction on composite dataset
    yhat = model.fit_predict(composite)
    # return just the predictions on the test set
    return yhat[len(trainX):]


def main():

    # models = ['LOF', 'EE', 'IF', 'SVM']
    targets = ['DOcategory', 'pHcategory']  # , 'ph', 'dissolved_oxygen']
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    # n_job = -1
    model, model_name = get_models()
    for j in range(len(model)):
        print(model_name[j])
        print(model[j])

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/AnomalyDetection/output_Cat_' + \
                    model_name[j]+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'std_test_score': 'std_test_score', 'mean_test_score': 'mean_test_score', 'params': 'params', 'bestscore': 'bestscore', 'fbeta': 'fbeta'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            resultFileName = 'results_'+target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directory+resultFileName,
                            index=False, header=False)

            path = 'Sondes_data/train/train_data/'
            method = 'SS_pipeline'

            for n_steps in [1, 3, 6, 12]:
                for PrH_index in [1, 3, 6, 12, 24, 36, 48]:
                    files = [f for f in os.listdir(path) if f.endswith(
                        '.csv') and f.startswith(sondefilename)]
                    file = files[0]
                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    dataset = pd.read_csv(path+file)
                    train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                        dataset, PrH_index, n_steps, target, cat)
                    print(train_X_grid[0:1])

                    start_time = time.time()

                    if model_name[j] == 'IF':
                        pipeline = Pipeline(steps=[('model', model[j])])
                    else:
                        pipeline = Pipeline(
                            steps=[('n', StandardScaler()), ('model', model[j])])

                    custom_cv = func.custom_cv_2folds(train_X_grid, 3)
                    i = 1
                    for train_index, test_index in custom_cv:
                        train_X_ = train_X_grid[train_index]
                        test_y_ = train_y_grid[train_index]
                        test_X = train_X_grid[test_index]
                        test_y = train_y_grid[test_index]

                        # fit on majority class
                        train_X_ = train_X_[test_y_ == 0]

                        # detect outliers in the test set
                        # if model_name[j] == 'LOF':
                        #     predictions = lof_predict(
                        #         model[j], train_X_, test_X)
                        # else:
                        pipeline.fit(train_X_)
                        predictions = pipeline.predict(test_X)

                        fpath = 'predictions_' + method+target+'_Window' + \
                            str(n_steps) + '_TH' + \
                            str(PrH_index)+'_CV' + str(i)+file

                        # mark inliers 1, outliers -1
                        test_y[test_y > 0] = -1
                        test_y[test_y == 0] = 1
                        # calculate score
                        score = f1_score(test_y, predictions, pos_label=-1)
                        print('F-measure: %.3f' % score)
                        # cm0 = predict(predictions, predictions, cat)

                        plt.scatter(np.arange(len(test_y)),
                                    test_y, s=1)
                        plt.scatter(np.arange(len(predictions)),
                                    predictions, s=1)
                        plt.legend(['actual', 'predictions'],
                                   loc='upper right')

                        plt.savefig(directory+fpath+'.jpg')

                        plt.close()

                        data = {'Actual': test_y, 'Predictions': predictions}
                        print(test_y.shape)
                        print(predictions.shape)

                        df = pd.DataFrame(data=data)
                        df.to_csv(directory+fpath, index=False)

                        if cat == 1:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': fpath, 'F-measure': score}

                        df = pd.DataFrame(data=data, index=[0])
                        df.to_csv(directory+resultFileName,
                                  index=False, mode='a', header=False)

                        elapsed_time = time.time() - start_time
                        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        i = i+1
                    Kb.clear_session()
                    gc.collect()
                    # del clf


if __name__ == "__main__":
    main()
