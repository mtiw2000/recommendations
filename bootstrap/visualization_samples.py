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
import seaborn as sns


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

budget = pd.read_csv("data/mn-budget-detail-2014.csv")
budget = budget.sort_values('amount',ascending=False)[:10]

pd.options.display.mpl_style = 'default'

budget_plot = budget.plot(kind="bar",x=budget["detail"],
                          title="MN Capital Budget - 2014",
                          legend=False)

fig = budget_plot.get_figure()
fig.savefig("data/2014-mn-capital-budget.png")

sns.set_style("darkgrid")

plt.xticks(rotation=90)

bar_plot = sns.barplot(x=budget["detail"],y=budget["amount"],palette="muted" )
plt.show()


sns.set_style("whitegrid")

ax = sns.stripplot(x=budget["detail"], y=budget["amount"],  jitter=True)
plt.show()

