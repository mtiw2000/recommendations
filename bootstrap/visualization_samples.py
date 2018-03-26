# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:07:34 2018
#https://mubaris.com/2017/09/26/introduction-to-data-visualizations-using-python/
#https://blog.modeanalytics.com/python-data-visualization-libraries/

@author: mtiw2
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['backend'] = "Qt4Agg"

os.getcwd()
os.chdir('C:/python_code/recommendations')


data = pd.read_csv('data/bitcoin_dataset.csv')
data.head()

a=data.describe()


data['Date'] = pd.to_datetime(data['Date'].values)

date = data['Date'].values
price = data['btc_market_price'].values




plt.plot(date, price)
plt.show()


# Random test data

salary = pd.read_csv('data/european_programmer_salary.csv')
salary.columns = ['Experience', 'Salary','Gender', 'Country']
salary.head()

salary = salary.groupby(['Country']).mean()
salary

country = salary.index[:5]
country_array = np.arange(5)
mean_salary = salary['Salary'].values[:5]

plt.bar()

plt.title("European Developers Salary")

plt.xticks(country_array, country)



# Y-Axis Label
plt.ylabel("Salary in Euro")

plt.bar(country_array, mean_salary, color='#f44c44')
        
        
        
# Basic Plot

# Title

# X-Axis Tick Labels

plt.show()

