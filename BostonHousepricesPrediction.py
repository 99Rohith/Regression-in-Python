#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:01:51 2020

@author: rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()

x_names = boston.feature_names

print(x_names)

x = boston.data
y = boston.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

print(reg.score(x_test,y_test))

