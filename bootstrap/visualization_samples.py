# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:07:34 2018
#https://mubaris.com/2017/09/26/introduction-to-data-visualizations-using-python/
#https://blog.modeanalytics.com/python-data-visualization-libraries/

@author: mtiw2
"""

#https://seaborn.pydata.org/generated/seaborn.boxplot.html
#https://bokeh.pydata.org/en/latest/docs/gallery.html#gallery
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from string import ascii_letters
import boto3
from  sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

from scipy import special

x=np.linspace(0,10,100)

dy = .8

y=np.sin(x) + dy * np.random.randn(100)

b = dy * np.random.randn(100)

plt.errorbar(x,y,yerr=dy,fmt='.k')



plt.style.use('classic')


plt.rcParams['backend'] = "Qt4Agg"
#print(plt.style.available)

os.getcwd()
os.chdir('C:/python_code/recommendations')

data = pd.read_csv('data/bitcoin_dataset.csv')
#data.head()

data['Date'] = pd.to_datetime(data['Date'].values)

date = data['Date'].values
price = data['btc_market_price'].values

plt.plot(date, price)
#plt.show()

np.empty(10)

x = np.linspace(0, 10, 30)

y=np.sin(x)

plt.plot(x, y, 'o', color='red');


plt.plot(x, y, '-ok');

plt.scatter(x, y, marker='o');



fig = plt.figure()
ax = plt.axes(projection='3d')


ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');



s
#box plot example 
sns.set_style("whitegrid")
tips = sns.load_dataset("tips")
ax = sns.boxplot(x=tips["total_bill"])

ax = sns.boxplot(x="day", y="total_bill", data=tips)

ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
                  data=tips, palette="Set3")


ax = sns.boxplot(x="day", y="total_bill", hue="time",
                 data=tips, linewidth=2.5)


tips["weekend"] = tips["day"].isin(["Sat", "Sun"])

 ax = sns.boxplot(x="day", y="total_bill", hue="weekend",
                  data=tips, dodge=False)


%timeit taxidata = pd.read_csv('data/nyctaxisub.csv')

with open('data/nyctaxisub.csv')  as f:
    reader = csv.reader()



# Random test data


np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

print (1.0/values)

big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)

%timeit (1.0/big_array)


x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))


x = np.arange(1, 6)
np.multiply.outer(x, x)



index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]

pop = pd.Series(populations, index=index)


index = pd.MultiIndex.from_tuples(index)
index

pop = pop.reindex(index)

pop[:,2010]




rng = np.random.RandomState(0)

x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)


plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

    
iris = load_iris()

features = iris.data.T

plt.scatter(features[0],features[1],alpha=0.2,s=100*features[3],c=iris.target,cmap='viridis')




x1=np.random.normal(0,.8,1000)
x2=np.random.normal(-2,1,1000)
x3=np.random.normal(3,2,1000)

kwargs = dict(histtype='stepfilled',alpha=0.3,normed=True,bins=40)

plt.hist(x1,**kwargs)
plt.hist(x2,**kwargs)
plt.hist(x3,**kwargs)




