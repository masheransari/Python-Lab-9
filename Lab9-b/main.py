import pandas as pd
import numpy as np
import sklearn.model_selection as train_test_split
import sklearn.svm as SVC
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("data.csv")
dataSet = pd.DataFrame(data)

print(dataSet.columns)

print(dataSet.info())

print(dataSet.head())

print(dataSet.describe().transpose())

dataSet.replace("?", np.nan, inplace=True)
funct = lambda x: (x.fillna(x.median()))
dataSet = dataSet.apply(funct, axis=1)

print(funct)
dataSet['Bare Nuclei'] = dataSet['Bare Nuclei'].astype('float64')
print(dataSet['Bare Nuclei'].astype('float64'))

target = dataSet['Class']
features = dataSet.drop(['ID', "Class"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)
svc_model = SVC(c=0.1, kernal="linear", gamma=1)
svc_model.fit(x_train, y_train)

predection = svc_model.predict(x_test)

print(svc_model.score(x_train, y_train))
print(svc_model.score(x_test, y_test))

print("Confusion Matrix:\n", confusion_matrix(predection, y_test))
