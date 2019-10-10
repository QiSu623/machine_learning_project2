import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.io import loadmat
from numpy.matlib import repmat
starttime = datetime.datetime.now()

def Gradient_Descent(m,c,learning_rate,iterate,x,y):
    n = float(len(x))
    for i in range(iterate):
        Y =m * x + c
        m -= learning_rate * ((-2 / n) * sum(x * (y - Y)))
        c -= learning_rate * ((-2 / n) * sum(y - Y))
    print(m,c)
    return m*x + c

# Making predictions
dataset = pd.read_csv("E:\Pythontest\kc_house_data.csv")
featurs_mean = list(dataset.columns[2:20])
corr = dataset[featurs_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
#plt.show()

space = dataset['sqft_above']
price = dataset['price']

x1 = np.array(space)
y1 = np.array(price)

x = (x1 - np.mean(x1)) / np.std(x1)
y = (y1 - np.mean(y1)) / np.std(y1)

Y_pred=Gradient_Descent(0, 0, 0.01, 100, x, y)

plt.scatter(x, y, color='green', edgecolors='red')
plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='blue')
#plt.show()
endtime = datetime.datetime.now()
print (endtime - starttime)




