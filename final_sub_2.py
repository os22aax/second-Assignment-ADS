# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:18:56 2022

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
df_urban_stat = df_urban[['1960','1970','1980','1990','2000','2010','2020']]
df_skew = stats.skew(df_urban_stat)
print('Skewness',df_skew)
#df_urban_stat = df_urban[['1960','1970','1980','1990','2000','2010','2020']]
df_kurtosis = stats.kurtosis(df_urban_stat)
print('kurtosis',df_kurtosis)

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
plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('Urban Population')
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
df_co2_stat = df_co2[['1960','1970','1980','1990','2000','2010']]
df_skew_co2 = stats.skew(df_co2_stat)
print('Skewness_co2',df_skew_co2)
#df_co2_stat = df_co2[['1960','1970','1980','1990','2000','2010']]
df_kurtosis_co2 = stats.kurtosis(df_co2_stat)
print('kurtosis_co2',df_kurtosis_co2)
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

plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('co2 emmission')
plt.legend()
plt.show()

#filter data from indicator name 'CO2 emissions from liquid fuel consumption'
df_agri = df_climate[df_climate["Indicator Name"] == 'Agricultural land (sq. km)']
#extract specific countries from dataframe
countries = ['China','India','United States','Norway','Finland']
df_agri = df_agri[df_agri['Country Name'].isin(countries)]
#remove 'nan' from colum and rows
df_agri = df_agri.reset_index().drop(['index'], axis=1)
df_agri = df_agri.drop(columns = ['Country Code','Indicator Name','Indicator Code'])
df_agri = df_agri.dropna(how='any', axis=1)
#reset the index
df_agri = df_agri.reset_index().drop(['index'], axis=1)
#extract specific colums
df_agri = df_agri[['Country Name','1970','1980','1990','2000','2010']]
print(df_agri)

#calculate both skewness and kurtisis values according to the years
df_agri_stat = df_agri[['1970','1980','1990','2000','2010']]
df_skew_agri = stats.skew(df_agri_stat)
print('Skewness_agri',df_skew_agri)
#df_agri_stat = df_agri[['1970','1980','1990','2000','2010']]
df_kurtosis_agri = stats.kurtosis(df_agri_stat)
print('kurtosis_agri',df_kurtosis_agri)
#calculate mean values country wise and yearly
df_mean_agri = df_agri.mean()
df_agri['Mean'] = df_agri.mean(axis=1)
print(df_agri)

#plot data in to the piechat (subplots)
plt.figure(figsize=(12,8))
plt.suptitle('Agricultural land', fontsize=20)
plt.subplot(3,2,1)
plt.pie( df_agri['1970'], labels = df_agri['Country Name'], autopct = '%1.2f%%', explode=(0.1,0.2,0.1,0.3,0.3))
plt.title('1970')
plt.subplot(3,2,2)
plt.pie( df_agri['1980'],labels = df_agri['Country Name'], autopct = '%1.2f%%', explode=(0.1,0.2,0.1,0.3,0.3))
plt.title('1980')
plt.subplot(3,2,3)
plt.pie( df_agri['1990'],labels = df_agri["Country Name"], autopct = '%1.2f%%', explode=(0.1,0.2,0.1,0.3,0.3))
plt.title('1990')
plt.subplot(3,2,4)
plt.pie( df_agri['2000'],labels = df_agri["Country Name"], autopct = '%1.2f%%', explode=(0.1,0.2,0.1,0.3,0.3))
plt.title('2000')
plt.subplot(3,2,5)
plt.pie( df_agri['2010'],labels = df_agri["Country Name"], autopct = '%1.2f%%', explode=(0.1,0.2,0.1,0.3,0.3))
plt.title('2010')
plt.show()

#filter data from indicator name 'CO2 emissions from liquid fuel consumption'
df_forest = df_climate[df_climate["Indicator Name"] == 'Forest area (sq. km)']
#extract specific countries from dataframe
countries = ['China','India','United States','Norway','Finland']
df_forest = df_forest[df_forest['Country Name'].isin(countries)]
#remove 'nan' from colum and rows
df_forest = df_forest.reset_index().drop(['index'], axis=1)
df_forest = df_forest.drop(columns = ['Country Code','Indicator Name','Indicator Code'])
df_forest = df_forest.dropna(how='any', axis=1)
#reset the index
df_forest = df_forest.reset_index().drop(['index'], axis=1)
#extract specific colums
df_forest = df_forest[['Country Name','1990','2000','2010','2020']]
print(df_forest)

#calculate both skewness and kurtisis values according to the years
df_forest_stat = df_forest[['1990','2000','2010','2020']]
df_skew_forest = stats.skew(df_forest_stat)
print('Skewness_co2',df_skew_forest)
#df_forest_stat = df_forest[['1990','2000','2010','2020']]
df_kurtosis_forest = stats.kurtosis(df_forest_stat)
print('kurtosis_co2',df_kurtosis_forest)
#calculate mean values country wise and yearly
df_mean_forest = df_forest.mean()
df_forest['Mean'] = df_forest.mean(axis=1)
print(df_forest)

#plot data in to the piechat (subplots)
plt.figure(figsize=(12,8))
plt.suptitle('Forest Area', fontsize=20)
plt.subplot(2,2,1)
plt.pie( df_forest['1990'], labels = df_forest['Country Name'], autopct = '%1.2f%%')
plt.title('1990')
plt.subplot(2,2,2)
plt.pie( df_forest['2000'],labels = df_forest['Country Name'], autopct = '%1.2f%%')
plt.title('2000')
plt.subplot(2,2,3)
plt.pie( df_forest['2010'],labels = df_forest["Country Name"], autopct = '%1.2f%%')
plt.title('2010')
plt.subplot(2,2,4)
plt.pie( df_forest['2020'],labels = df_forest["Country Name"], autopct = '%1.2f%%')
plt.title('2020')
plt.show()