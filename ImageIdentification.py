import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
dataset= fetch_mldata('MNIST original')

X=dataset.data
y=dataset.target

some_digit=X[65000]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image)
plt.show()

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X,y)
nb.score(X,y)
nb.predict(X[[65000,3876,19,40],0:784])

from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=9)
dtf.fit(X,y)
dtf.score(X,y)
dtf.predict(X[[65000,3876,19,40],0:784])
