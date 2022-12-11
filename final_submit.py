# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 03:52:15 2022

@author: Omesha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

print(os.getcwd())
files = os.listdir()
print(files)


def pass_file(file):
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
    for file in files:
        if os.path.splitext(file)[1]=='.csv':
            df_climate = pd.read_csv(file, skiprows=4)
            df_climate_t = pd.DataFrame.transpose(df_climate)
            df_climate_t_h = df_climate_t.iloc[0].values.tolist()
            df_climate_t.columns = df_climate_t_h
            df_climate_new = df_climate_t.iloc[4:]
            
            return(df_climate,df_climate_new)
        
#import climate change 'csv' file        
df_climate, df_climate_new = pass_file("climate_change.csv") 

#filter data from indicator name 'urban population'
df_urban = df_climate[df_climate["Indicator Name"] == 'Urban population']
#extract specific countries from dataframe
countries = ['China','India','United States','Norway','Finland']
df_urban = df_urban[df_urban['Country Name'].isin(countries)]
#remove 'nan' from colum and rows
df_urban = df_urban.reset_index().drop(['index'], axis=1)
df_urban = df_urban.drop(columns = ['Country Code','Indicator Name','Indicator Code'])
df_urban = df_urban.dropna(how='any', axis=1)
#reset the index
df_urban = df_urban.reset_index().drop(['index'], axis=1)
#extract specific colums
df_urban_graph = df_urban[['Country Name','1960','1970','1980','1990','2000','2010','2020']]

#calculate both skewness and kurtisis values according to the years
df_skew_1960 = stats.skew(df_urban_graph['1960'])
df_kurtosis_1960 = stats.kurtosis(df_urban_graph['1960'])
df_skew_1970 = stats.skew(df_urban_graph['1970'])
df_kurtosis_1970 = stats.kurtosis(df_urban_graph['1970'])
df_skew_1980 = stats.skew(df_urban_graph['1980'])
df_kurtosis_1980 = stats.kurtosis(df_urban_graph['1980'])
df_skew_1990 = stats.skew(df_urban_graph['1990'])
df_kurtosis_1990 = stats.kurtosis(df_urban_graph['1990'])
df_skew_2000 = stats.skew(df_urban_graph['2000'])
df_kurtosis_2000 = stats.kurtosis(df_urban_graph['2000'])
df_skew_2010 = stats.skew(df_urban_graph['2010'])
df_kurtosis_2010 = stats.kurtosis(df_urban_graph['2010'])
df_skew_2020 = stats.skew(df_urban_graph['2020'])
df_kurtosis_2020 = stats.kurtosis(df_urban_graph['2020'])

print(df_skew_1960, df_skew_1970, df_skew_1980, df_skew_1990, 
      df_skew_2000, df_skew_2010, df_skew_2020)
print(df_kurtosis_1960, df_kurtosis_1970, df_kurtosis_1980, df_kurtosis_1990,
      df_kurtosis_2000, df_kurtosis_2010, df_kurtosis_2020 )

#calculate mean values country wise and yearly
df_mean = df_urban_graph.mean()
df_urban_graph['Mean'] = df_urban_graph.mean(axis=1)
print(df_urban_graph)

#plot data in to the line graph
plt.figure()
plt.plot(df_urban_graph['Country Name'], df_urban_graph['1960'], label="1960")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['1970'], label="1970")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['1980'], label="1980")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['1990'], label="1990")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['2000'], label="2000")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['2010'], label="2010")
plt.plot(df_urban_graph['Country Name'], df_urban_graph['2020'], label="2020")
#plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('Urban Population')
plt.legend()
plt.show()

#plot data in to the histogram (subplots)
plt.figure(figsize=(12,8))
plt.subplot(4,2,1)
plt.hist( df_urban_graph['1960'],label="1960")
plt.legend()
plt.subplot(4,2,2)
plt.hist( df_urban_graph['1970'],label="1970")
plt.legend()
plt.subplot(4,2,3)
plt.hist( df_urban_graph['1980'],label="1980")
plt.legend()
plt.subplot(4,2,4)
plt.hist( df_urban_graph['1990'],label="1990")
plt.legend()
plt.subplot(4,2,5)
plt.hist( df_urban_graph['2000'],label="2000")
plt.legend()
plt.subplot(4,2,6)
plt.hist( df_urban_graph['2010'],label="2010")
plt.legend()
plt.subplot(4,2,7)
plt.hist( df_urban_graph['2020'],label="2020")
plt.legend()
plt.show()

#filter data from indicator name 'CO2 emissions from liquid fuel consumption'
df_co2 = df_climate[df_climate["Indicator Name"] == 'CO2 emissions from liquid fuel consumption (kt)']
#extract specific countries from dataframe
countries = ['China','India','United States','Norway','Finland']
df_co2 = df_co2[df_co2['Country Name'].isin(countries)]
#remove 'nan' from colum and rows
df_co2 = df_co2.reset_index().drop(['index'], axis=1)
df_co2 = df_co2.drop(columns = ['Country Code','Indicator Name','Indicator Code'])
df_co2 = df_co2.dropna(how='any', axis=1)
#reset the index
df_co2 = df_co2.reset_index().drop(['index'], axis=1)
#extract specific colums
df_co2 = df_co2[['Country Name','1960','1970','1980','1990','2000','2010']]
print(df_co2)

#calculate both skewness and kurtisis values according to the years
df_skew_1960_co2 = stats.skew(df_co2['1960'])
df_kurtosis_1960_co2 = stats.kurtosis(df_co2['1960'])
df_skew_1970_co2 = stats.skew(df_co2['1970'])
df_kurtosis_1970_co2 = stats.kurtosis(df_co2['1970'])
df_skew_1980_co2 = stats.skew(df_co2['1980'])
df_kurtosis_1980_co2 = stats.kurtosis(df_co2['1980'])
df_skew_1990_co2 = stats.skew(df_co2['1990'])
df_kurtosis_1990_co2 = stats.kurtosis(df_co2['1990'])
df_skew_2000_co2 = stats.skew(df_co2['2000'])
df_kurtosis_2000_co2 = stats.kurtosis(df_co2['2000'])
df_skew_2010_co2 = stats.skew(df_co2['2010'])
df_kurtosis_2010_co2 = stats.kurtosis(df_co2['2010'])

#calculate mean values country wise and yearly
df_mean_co2 = df_co2.mean()
df_co2['Mean'] = df_co2.mean(axis=1)
print(df_co2)

#plot data in to the line graph
plt.figure()
plt.plot(df_co2['Country Name'], df_co2['1960'], label="1960")
plt.plot(df_co2['Country Name'], df_co2['1970'], label="1970")
plt.plot(df_co2['Country Name'], df_co2['1980'], label="1980")
plt.plot(df_co2['Country Name'], df_co2['1990'], label="1990")
plt.plot(df_co2['Country Name'], df_co2['2000'], label="2000")
plt.plot(df_co2['Country Name'], df_co2['2010'], label="2010")

#plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('co2 emmission')
plt.legend()
plt.show()

#plot data in to the histogram (subplots)
plt.figure(figsize=(12,8))
plt.subplot(4,2,1)
plt.hist( df_co2['1960'],label="1960")
plt.legend()
plt.subplot(4,2,2)
plt.hist( df_co2['1970'],label="1970")
plt.legend()
plt.subplot(4,2,3)
plt.hist( df_co2['1980'],label="1980")
plt.legend()
plt.subplot(4,2,4)
plt.hist( df_co2['1990'],label="1990")
plt.legend()
plt.subplot(4,2,5)
plt.hist( df_co2['2000'],label="2000")
plt.legend()
plt.subplot(4,2,6)
plt.hist( df_co2['2010'],label="2010")
plt.legend()
plt.show()


