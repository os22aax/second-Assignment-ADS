# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:09:25 2022

@author: Omesha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

def parse_file(file):
    '''
    import data file in world bank format and transpose the import data file 
    and have two dataframes as output both original and transposed
    Parameters
    ----------
    file : imported 'csv' file from current working directory 
    

    Returns
    -------
    df_climate : imported original dataframe
    df_climate_new : transposed dataframe

    '''

    print(os.getcwd())
    files = os.listdir()
    print(files)

    for file in files:
        if os.path.splitext(file)[1]=='.csv':
            df_climate = pd.read_csv(file, skiprows=4, index_col = 'Country Name')
            df_climate_t = pd.DataFrame.transpose(df_climate)
            df_climate_t_h = df_climate_t.iloc[0].values.tolist()
            df_climate_t.columns = df_climate_t_h
            df_climate_new = df_climate_t.iloc[4:]
            
            return(df_climate,df_climate_new)
        
#import climate change 'csv' file        
df_climate, df_climate_new = parse_file("climate_change.csv")         
        
#select population and solid fuel CO2 emissions from main data
df_population = df_climate[df_climate["Indicator Code"].isin([ "SP.URB.TOTL"])]
df_co2_emissions = df_climate[df_climate["Indicator Code"].isin(["EN.ATM.CO2E.LF.KT"])]

#drop columns that are not needed
df_co2_emissions = df_co2_emissions.drop(columns = ['Country Code','Indicator Name','Indicator Code'])
df_population = df_population.drop(columns = ['Country Code','Indicator Name','Indicator Code'])

#make a list of countries of interest
countries = ['China','India','United States','Norway','Finland']

#filter by the above countries list
df_co2_emissions = df_co2_emissions.filter(items = countries, axis = 0)
df_population = df_population.filter(items = countries, axis = 0)
df_co2_emissions = df_co2_emissions.dropna(how='any', axis=1)
df_population = df_population.dropna(how='any', axis=1)

#calculate both skewness and kurtisis values according to the years
df_co2_stat = df_co2_emissions[['1960','1970','1980','1990','2000','2010']]
df_skew = stats.skew(df_co2_stat)
print('Skewness',df_skew)
df_kurtosis = stats.kurtosis(df_co2_stat)
print('kurtosis',df_kurtosis)

#calculate mean values 
df_mean = df_co2_emissions.mean()

df_population_stat = df_population[['1960','1970','1980','1990','2000','2010','2020']]
df_skew = stats.skew(df_population_stat)
print('Skewness',df_skew)
df_kurtosis = stats.kurtosis(df_population_stat)
print('kurtosis',df_kurtosis)

#calculate mean values 
df_mean = df_co2_emissions.mean()
   
print(df_co2_emissions) 
print(df_population)

#transpose in order be able to plot as a time series    
df_co2_emissions_1 = df_co2_emissions.transpose()   
df_population = df_population.transpose() 
print(df_co2_emissions_1) 
    
plt.figure()
df_co2_emissions_1.plot()
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('CO2 emissions from liquid fuel consumption')
plt.legend()
plt.show()

plt.figure()
df_population.plot()
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Urban Population')
plt.legend()
plt.show()







