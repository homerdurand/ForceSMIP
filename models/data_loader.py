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

def load_model_data(model, var, path='../data/'):
    # Loading data file
    file_path = os.path.join(path, '{}_{}_monthly.nc'.format(model, var))
    ds = xr.open_dataset(file_path)
    # Getting TAS
    tas_array = ds[var].values
    # Close the dataset
    ds.close()
    return tas_array

def get_data_shape_lat_lon(model='CanESM5', var='tas', path='data/'):
    file_path = os.path.join(path, '{}_{}_monthly.nc'.format(model, var))
    ds = xr.open_dataset(file_path)
    # Getting TAS
    shape = ds[var].values.shape
    # Close the dataset
    ds.close()
    return shape, ds['lat'], ds['lon']

def load_data_models(models, var='tas', n_sample=2500, path='./data/', coarse=24):
    X, y = None, None
    for model in models:
        tas_array = load_model_data(model, var=var, path=path)
        tas_array = np.transpose(tas_array, axes=(1, 2, 0, 3, 4))
        shape = tas_array.shape
        
        idxs = random.sample(range(shape[1]*shape[2]), n_sample)
        X_temp = tas_array.reshape(shape[0], shape[1]*shape[2], shape[3]//coarse, coarse, shape[4]//coarse, coarse).mean(axis=(3,5)).reshape(shape[0], shape[1]*shape[2], shape[3]*shape[4]//(coarse**2))
        y_temp = np.tile(tas_array.mean(axis=2), (shape[2], 1, 1)) .reshape(shape[0], shape[1]*shape[2], shape[3]*shape[4])
        if X is None:
            X = X_temp[:,idxs,:]
            y = y_temp[:,idxs,:]
        else :
            X = np.stack((X, X_temp[:,idxs,:]), axis=1)
            y = np.stack((y, y_temp[:,idxs,:]), axis=1)
        del tas_array
        del X_temp
        del y_temp
    return X, y