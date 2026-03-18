#%%%
import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    
    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% 1 -  2mL_sumergida_simple'

paths_1 = glob('1_2mL_sumergida_simple/**/*.csv', recursive=True)
# %%
for p in paths_1:       
    name=os.path.basename(p)[14:-4]
    A,t,T,B= lector_templog(p)
    
    plt.figure(figsize=(8,4),constrained_layout=True)
    plt.plot(t,T,'.-')
    plt.axhline(y=0, color='k')
    
    plt.legend()
    plt.title(name)
    plt.ylabel('T (°C)')
    plt.xlabel('t (s)')
    plt.grid()      
    plt.show()

#%%
