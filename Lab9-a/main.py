# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import pandas as pd
# from sklearn.datasets import load_iris
# iris = load_iris()
#
# dir(iris)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()


# print(iris.feature_names)
#
# print(iris.target_names)
#
# print(iris.data)
#
# print(len(iris.data))
#
#
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
#
df['target'] = iris.target
# print(df.head())
#
# print(iris.target)
#
# print(df[df.target == 2].head())
#
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
# print(df.head())

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# %matplotlib inline

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="blue",marker=',')


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color="blue",marker=',')




X = df.drop(['target','flower_name'],axis='columns')
y = df.target


print(X)
print(y)

from sklearn.model_selection import train_test_split
x = df.drop(['target','flower_name'],axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print(len(X_train))
print(len(X_test))

from sklearn.svm import SVC

model = SVC()
print(model.fit(X_train,y_train))

print(model.predict([[6.4,3.2,4.5,1.5]]))

model_C = SVC(C=10)
model_C.fit(X_train,y_train)
print(model_C.score(X_test,y_test))

model_g = SVC(gamma=10)
model_g.fit(X_train,y_train)
print(model_g.score(X_test,y_test))

model_linear_kernal = SVC(kernel='linear')
print(model_linear_kernal.fit(X_train,y_train))
print(model_linear_kernal.score(X_test,y_test))





