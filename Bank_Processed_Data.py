import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Bank.csv', sep=';')

data.info()
data.isnull().sum()

X = data.iloc[:, 0:16].values
y = data.iloc[:,-1].values 

X=pd.DataFrame(X)
y=pd.DataFrame(y)

#To conver the string columns to numerical
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
for i in [1,2,3,4,6,7,8,10,15]:
    X.iloc[:,i]=lab.fit_transform(X.iloc[:, i])
y=lab.fit_transform(y)

#to remove the dummy creation
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[1,2,3,4,6,7,8,10,15])
X=one.fit_transform(X)
X=X.toarray()

#to scale the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
l_r=LogisticRegression()
l_r.fit(X,y)
l_r.score(X,y)

y_pred=l_r.predict(X)

# Confusion Matrix for Logistic Regression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm=confusion_matrix(y,y_pred)
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=2)
dtf.fit(X,y)
dtf.score(X,y)

y_pred1=dtf.predict(X)

# Confusion Matrix for Decision Tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm2=confusion_matrix(y,y_pred1)
precision_score(y,y_pred1)
recall_score(y,y_pred1)
f1_score(y,y_pred1)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X,y)
knn.score(X,y)

#Naive-Bayes
from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()
n_b.fit(X,y)
n_b.score(X,y)




