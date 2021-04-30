# -*- coding: utf-8 -*-
"""
                               --README--
                                
The Code Is Developed to be Modular, easy to extend, parameterizable.
The Code can be simply used to Plot Time Series and is  Fully compatible with Vigcrues Website
,Any ".csv" data File Downloaded from the Vigcrues website is Fully Compatible with the code.
Comments are included in the code that are self explanatory about the working of that part of code. 

the code is divided into cells to make it easy to understand . 

Cell 1 - It contains the Function required to calculate the Stationarity.
Cell 2.3 - These Cells Contain Plotting the Graphs for Nartuby River -Level and Speed Separately
Cell 4,5 - These Cells Contain Plotting the Graphs for La Pique River -Level and Speed Separately
Cell 6,7 - These Cells Contain Predictions for Nartuby River -Level and Speed Separately
Cell 8,9 - These Cells Contain Predictions for La Pique River -Level and Speed Separately

Code is Developed By Pamit DUGGAL 
Under The Guidance of Prof. Jean Marc PIERSON

"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import *
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf


import datetime as dt

#%% 
    #Cell 1
#Function for testing stationarity of time series plot
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(15).mean()
    rolstd = timeseries.rolling(15).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
#%% 
    #Cell 2
#Fetching csv file for Nartuby Level
Nartuby_Level = pd.read_csv('Vigicrues_Hauteurs_Y523501001-Nartuby.csv')
print(Nartuby_Level.head())
print('\n Nartuby_Level Types:')
s = 'Date et heure locale;"Trans-en-Provence [CD 555] [Décathlon] (Nartuby) (m)"'

#Splitting one column into two
new = Nartuby_Level[s].str.split(";", n = 1, expand = True) 
Nartuby_Level["Date and Time"]= new[0] 
Nartuby_Level["Water Level(m)"]= new[1] 
Nartuby_Level.drop(columns =[s], inplace = True) 
print(Nartuby_Level)

#After installing pip datetime to convert to time series
Nartuby_Level['Date and Time']=pd.to_datetime(Nartuby_Level['Date and Time'])
#Seperating only time from date and time
Nartuby_Level['Date and Time'] = Nartuby_Level['Date and Time'].dt.strftime('%H:%M')
Nartuby_Level.set_index('Date and Time', inplace=True)
#check datatype of index
print(Nartuby_Level.index)

#ts signifies y-axis values on plot
ts = Nartuby_Level['Water Level(m)'].astype(float)

#plotting time series of Nartuby level
print(ts)
plt.plot(ts)
plt.tight_layout()
plt.title('Nartuby Level')

plt.xticks(rotation=90)
plt.xlabel("Time")
plt.ylabel("Water Level (m)")

#adjusting plot dimensions
plt.rc('xtick',labelsize=7.5)
plt.rcParams["figure.figsize"] = (8,3)

plt.show()
#converting linear scale to logarithmic scale for prediction with respect to time
ts_log = np.log(ts)
#Testing stationarity
test_stationarity(ts)
ts_log_diff = ts_log - ts_log.shift()
plt.title('Nartuby Level Logarithmic')
plt.plot(ts_log_diff)


#%%
    #Cell 3
#Fetching csv file for Nartuby Speed
Nartuby_Speed = pd.read_csv('Vigicrues_Debits_Y523501001-Nartuby.csv')
print(Nartuby_Speed.head())
print('\n Nartuby_Speed Types:')
s1 = 'Date et heure locale;"Trans-en-Provence [CD 555] [Décathlon] (Nartuby) (m³/s)"'
#Splitting one column into two
new = Nartuby_Speed[s1].str.split(";", n = 1, expand = True) 
Nartuby_Speed["Date and Time"]= new[0] 
Nartuby_Speed["Flow(m3/s)"]= new[1] 
Nartuby_Speed.drop(columns =[s1], inplace = True) 
print(Nartuby_Speed)
#convert to time series
Nartuby_Speed['Date and Time']=pd.to_datetime(Nartuby_Speed['Date and Time'])
Nartuby_Speed['Date and Time'] = Nartuby_Speed['Date and Time'].dt.strftime('%H:%M')
Nartuby_Speed.set_index('Date and Time', inplace=True)
#check datatype of index
print(Nartuby_Speed.index)
#ts1 signifies y-axis values on plot
ts1 = Nartuby_Speed['Flow(m3/s)'].astype(float)

#plotting time series of Nartuby Speed
print(ts1)
plt.plot(ts1)
plt.tight_layout()
plt.title('Nartuby Speed')

plt.xticks(rotation=90)
plt.xlabel("Time")
plt.ylabel("Flow(m3/s)")
#adjusting plot dimensions
plt.rc('xtick',labelsize=7.5)
plt.rc('xtick',labelsize=8)
plt.rcParams["figure.figsize"] = (10,4)

plt.show()
#converting linear scale to logarithmic scale for prediction with respect to time
ts1_log = np.log(ts1)
#Testing stationarity
test_stationarity(ts1)
ts1_log_diff = ts1_log - ts1_log.shift()
plt.title('Nartuby Speed Logarithmic')
plt.plot(ts1_log_diff)
#%% 
   #Cell 4
#Fetching csv file for Pique Level
Pique_Level = pd.read_csv('Vigicrues_Hauteurs_O004402001-Pique.csv')
print(Pique_Level.head())
print('\n Pique_Level Types:')
s2 = 'Date et heure locale;"Bagnères-de-Luchon (Pique) (m)"'
#Splitting one column into two
new = Pique_Level[s2].str.split(";", n = 1, expand = True) 
Pique_Level["Date and Time"]= new[0] 
Pique_Level["Water Level(m)"]= new[1] 
Pique_Level.drop(columns =[s2], inplace = True) 
print(Pique_Level)
#convert to time series
Pique_Level['Date and Time']=pd.to_datetime(Pique_Level['Date and Time'])
Pique_Level['Date and Time'] = Pique_Level['Date and Time'].dt.strftime('%H:%M')
Pique_Level.set_index('Date and Time', inplace=True)
#check datatype of index
print(Pique_Level.index)
#ts2 signifies y-axis values on plot
ts2 = Pique_Level['Water Level(m)'].astype(float)
#plotting time series of Pique Level
print(ts2)
plt.plot(ts2)
plt.tight_layout()
plt.title('Pique Level')

plt.xticks(rotation=90)
plt.xlabel("Time")
plt.ylabel("Water Level (m)")
#adjusting plot dimensions
plt.rc('xtick',labelsize=7.5)
plt.rc('ytick',labelsize=7.0)

plt.rcParams["figure.figsize"] = (8,3)

plt.show()
#converting linear scale to logarithmic scale for prediction with respect to time
ts2_log = np.log(ts2)
#Testing stationarity
test_stationarity(ts2)
ts2_log_diff = ts2_log - ts2_log.shift()
plt.title('Pique Level Logarithmic')
plt.plot(ts2_log_diff)
#%% 
    #Cell 5
#Fetching csv file for Pique Speed
Pique_Speed = pd.read_csv('Vigicrues_Debits_O004402001-Pique.csv')
print(Pique_Speed.head())
print('\n Pique_Speed Types:')
s3 = 'Date et heure locale;"Bagnères-de-Luchon (Pique) (m³/s)"'
#Splitting one column into two
new = Pique_Speed[s3].str.split(";", n = 1, expand = True) 
Pique_Speed["Date and Time"]= new[0] 
Pique_Speed["Flow(m3/s)"]= new[1] 
Pique_Speed.drop(columns =[s3], inplace = True) 
print(Pique_Speed)
#convert to time series
Pique_Speed['Date and Time']=pd.to_datetime(Pique_Speed['Date and Time'])
Pique_Speed['Date and Time'] = Pique_Speed['Date and Time'].dt.strftime('%H:%M')
Pique_Speed.set_index('Date and Time', inplace=True)
#check datatype of index
print(Pique_Speed.index)
#ts3 signifies y-axis values on plot
ts3 = Pique_Speed['Flow(m3/s)'].astype(float)

#plotting time series of Pique Speed
print(ts3)
plt.plot(ts3)
plt.tight_layout()
plt.title('Pique Speed')

plt.xticks(rotation=90)
plt.xlabel("Time")
plt.ylabel("Flow(m3/s)")
#adjusting plot dimensions
plt.rc('xtick',labelsize=7.5)
plt.rc('xtick',labelsize=8)
plt.rcParams["figure.figsize"] = (10,4)
plt.show()
#converting linear scale to logarithmic scale for prediction with respect to time
ts3_log = np.log(ts3)
#Testing stationarity
test_stationarity(ts3)
ts3_log_diff = ts3_log - ts3_log.shift()
plt.title('Pique Speed Logarithmic')
plt.plot(ts3_log_diff)

#%%
    #Cell 6
#Predicting values using ARIMA Model for Nartuby Level
model = ARIMA(ts_log, order = (2,1,2))
results_ARIMA = model.fit(disp=1)
#Making predictions on logarithmic scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff.cumsum())
predictions_ARIMA_log = pd.Series(ts_log.iloc[0],index = ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#Converting back to linear scale for comparison
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#Plotting the prediction plot vs original plot in linear scale
plt.plot(ts)
plt.plot(predictions_ARIMA)
#Calculating the RMSE (Root Mean Square Error)
plt.title('Nartuby Level RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
#%%  
    #Cell 7
#Predicting values using ARIMA Model for Nartuby Speed
model = ARIMA(ts1_log, order = (2,1,2))
results_ARIMA = model.fit(disp=1)
#Making predictions on logarithmic scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff.cumsum())
predictions_ARIMA_log = pd.Series(ts1_log.iloc[0],index = ts1_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#Converting back to linear scale for comparison
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#Plotting the prediction plot vs original plot in linear scale
plt.plot(ts1)
plt.plot(predictions_ARIMA)
#Calculating the RMSE (Root Mean Square Error)
plt.title('Nartuby Speed RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts1)**2)/len(ts1)))
#%%
    #Cell 8
#Predicting values using ARIMA Model for Pique Level
model = ARIMA(ts2_log, order = (2,1,2))
results_ARIMA = model.fit(disp=1)
#Making predictions on logarithmic scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff.cumsum())
predictions_ARIMA_log = pd.Series(ts2_log.iloc[0],index = ts2_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#Converting back to linear scale for comparison
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#Plotting the prediction plot vs original plot in linear scale
plt.plot(ts2)
plt.plot(predictions_ARIMA)
#Calculating the RMSE (Root Mean Square Error)
plt.title('Pique Level RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts2)**2)/len(ts2)))

#%%
    #Cell 9
#Predicting values using ARIMA Model for Pique Speed
model = ARIMA(ts3_log, order = (2,1,2))
results_ARIMA = model.fit(disp=1)
#Making predictions on logarithmic scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff.cumsum())
predictions_ARIMA_log = pd.Series(ts3_log.iloc[0],index = ts3_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#Converting back to linear scale for comparison
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#Plotting the prediction plot vs original plot in linear scale
plt.plot(ts3)
plt.plot(predictions_ARIMA)
#Calculating the RMSE (Root Mean Square Error)
plt.title('Pique Speed RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts3)**2)/len(ts3)))

#End of Code


