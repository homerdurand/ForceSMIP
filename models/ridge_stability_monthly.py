from matplotlib import pyplot as plt

import xarray as xr
import netCDF4 as nc
import numpy as np

import os

import datetime as dt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import random
from tqdm import tqdm
from collections import Counter
from data_loader import *

import time


models = ['CanESM5', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MIROC6', 'CESM2']


class RidgeCV_monthly:
    def __init__(self, alphas=None, cv=2) -> None:
        if alphas is None:
            alphas = np.logspace(1, 3, 3)
        self.alphas = alphas
        self.cv = cv
    
    def fit(self, X, Y, refit=False):
        if not refit:
            self.coefs_ = []
            self.intercepts_ = []
            self.n_train = X.shape[1]
        ridge = RidgeCV(alphas=self.alphas, cv=self.cv)
        for i in range(X.shape[0]):
            ridge.fit(X[i,:,:], Y[i,:,:])
            if not refit:
                self.coefs_.append(ridge.coef_)
                self.intercepts_.append(ridge.intercept_)
                self.n_train = X.shape[1]
            else :
                self.coefs_[i] = (self.n_train*self.coefs_[i] + X.shape[1]*ridge.coef_)/(self.n_train + X.shape[1])
                self.intercepts_[i] = (self.n_train*self.intercepts_[i] + X.shape[1]*ridge.intercept_)/(self.n_train + X.shape[1])
                self.n_train += X.shape[1]
                
    def predict(self, X):
        Y_pred = []
        for i in range(X.shape[0]):
            y_pred = X[i,:,:] @ self.coefs_[i].T + self.intercepts_[i]
            Y_pred.append(y_pred)
        return np.array(Y_pred)
    
    def score(self, X, Y, average=False, multioutput='uniform_average'):
        Y_pred = self.predict(X)
        scores = [r2_score(Y[i,:,:], Y_pred[i,:,:], multioutput=multioutput) for i in range(X.shape[0])]
        if average:
            return np.mean(scores) 
        else :
            return scores
        
        
print('Loading training data...', end='\r')
X_train, Y_train = load_data_models(['CESM2'], var='tas', n_sample=2000, path='data/')
shape = X_train.shape
X_train, Y_train = X_train.reshape(shape[0]*shape[1], -1), Y_train.reshape(shape[0]*shape[1], -1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
print('Data training loaded!')
print(X_train.shape)
print('Training Ridge...', end='\r')
# # # ridge = RidgeCV_monthly()
# # # ridge.fit(X_train, Y_train)
# n_alpha = 5
# alphas = np.logspace(6, 9, n_alpha)
# List of solvers to compare

ridge = Ridge(alpha=1.0, solver='svd')
ridge.fit(X_train, Y_train)


print('Loading data train2...', end='\r')
X_test, Y_test = load_data_models(['CanESM5'], var='tas', n_sample=2000, path='data/')
X_test, Y_test = X_test.reshape(shape[0]*shape[1], -1), Y_test.reshape(shape[0]*shape[1], -1)
print('Data train2 loaded!')


print('Computing scores...', end='\r')
score1 = ridge.score(X_train, Y_train)
score2 = ridge.score(X_val, Y_val)
score3 = ridge.score(X_test, Y_test )
print('Scores computed!')

print(score1, score2, score3)
# print(ridge.alpha_)