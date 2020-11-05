#!/usr/bin/python3
from sklearn.linear_model import LogisticRegression
import pandas as pd
from icecream import ic
import sys

train = pd.read_csv(sys.argv[1], header=None).values
test = pd.read_csv(sys.argv[2], header=None).values
y_train, X_train = train[:,0], train[:,1:]
y_test, X_test = test[:,0], test[:,1:]
ic(y_train, X_train)
clf = LogisticRegression(fit_intercept=False, C = 1e15)
clf.fit(X_train, y_train)
ic(clf.coef_)
ic(clf.score(X_test, y_test))