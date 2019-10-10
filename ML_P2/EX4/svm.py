import pandas as pd
import numpy as np
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime

starttime = datetime.datetime.now()
# display setting
pd.set_option('display.width',None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

# datasets setting and read
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']
data = pd.read_csv("E:\Pythontest\wisconsin.data", names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
print(data.shape)
print(data.head())

# query Benign and Malignant data
sns.countplot(data['Class'], label='Count')

# heatmap
featurs_mean = list(data.columns[1:11])
corr = data[featurs_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:9]],data[column_names[10]],test_size=0.2, random_state=5)
model = svm.SVC(C=0.1, kernel='poly', degree=1, gamma='scale')
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('accuracy:', metrics.accuracy_score(prediction, y_test))


mtrx = confusion_matrix(y_test, prediction)
np.set_printoptions(precision=2)
plot_confusion_matrix(mtrx)
#plt.show()
endtime = datetime.datetime.now()
print (endtime - starttime)
