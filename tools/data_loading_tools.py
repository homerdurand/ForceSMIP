from matplotlib import pyplot as plt

import xarray as xr
import netCDF4 as nc
import numpy as np

import os

import datetime as dt
import numpy as np

def load_data_monthly(models, var='tas', path='../AnchorMultivariateAnalysis/data/ForceSMIP/Training/Amon/tas/ForceSMIP/', coarsen_factor=None):
    ensemble = {}
    flag = True
    for model in models:
        print('## Model {}'.format(model))
        directory = path + model
        listdir = os.listdir(directory)
        
        data = None
        ensemble[model] = {}
        for i, file in enumerate(listdir, start=1):
            if i>25:
                break
            print('File {}/{}'.format(i, len(listdir)), end='\r')
            file_path = os.path.join(directory, file)
            ds = xr.open_dataset(file_path)
            
            if flag:
                ensemble['time.month'] = ds['time.month']
                ensemble['time.year'] = np.unique(ds['time.year'])
                ensemble['lat'] = ds['lat'].values
                ensemble['lon'] = ds['lon'].values  
                flag = False
                
            if data is None:
                data = [ np.transpose(ds[var].values.reshape((-1, 12, 72, 144)), (1, 0, 2, 3)) ]
            else :
                data.append( np.transpose(ds[var].values.reshape((-1, 12, 72, 144)), (1, 0, 2, 3)) )
            ds.close()
        print()
        ensemble[model][var] = np.array(data)
    return ensemble



def save_data(data, model='CanESM5', var='tas', data_path='data/'):
    n_members = data[model][var].shape[0]
    n_months = data[model][var].shape[1]
    n_years = len(data['time.year'])
    
    # Create a NetCDF file
    with nc.Dataset(data_path + '{}_{}_monthly.nc'.format(model, var), 'w') as f:
        # Define dimensions
        f.createDimension('n_members', n_members)
        f.createDimension('n_months', n_months)
        f.createDimension('n_years', n_years)
        f.createDimension('lat', len(data['lat']))
        f.createDimension('lon', len(data['lon']))

        # Create variables
        members_var = f.createVariable('n_members', 'i4', ('n_members',))
        months_var = f.createVariable('n_months', 'i4', ('n_months',))
        years_var = f.createVariable('n_years', 'i4', ('n_years',))
        lat_var = f.createVariable('lat', 'f4', ('lat',))
        lon_var = f.createVariable('lon', 'f4', ('lon',))
        tas_var = f.createVariable('tas', 'f4', ('n_members', 'n_months', 'n_years', 'lat', 'lon'))

        # Assign data to variables
        members_var[:] = np.arange(n_members)
        months_var[:] = np.arange(1, n_months + 1)
        years_var[:] = data['time.year']
        lat_var[:] = data['lat']
        lon_var[:] = data['lon']
        tas_var[:] = data[model][var]

        # Add attributes if necessary
        members_var.units = 'member index'
        months_var.units = 'month index'
        years_var.units = 'year index'
        lat_var.units = 'latitude units'
        lon_var.units = 'longitude units'
        tas_var.units = 'temperature units'

        # Add additional metadata if needed
        tas_var.description = 'Surface air temperature'


models = ['MIROC6', 'CESM2', 'CanESM5', 'MIROC-ES2L', 'MPI-ESM1-2-LR']


for model in models :
    data = load_data_monthly([model])
    save_data(data, model=model, var='tas')