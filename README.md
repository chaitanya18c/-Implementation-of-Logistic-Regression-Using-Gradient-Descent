# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary.

4.Calculate the y-prediction.


## Program:
``` python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: CHAITANYA P S
RegisterNumber: 212222230024
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Y_PRED:",y_pred)
print("Y:",Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)


```

## Output:
#### Dataset:
![326149135-f0e5af66-87dc-4dd7-be43-c817ad0d6473](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/19fcc04d-605d-4ed6-9180-b56112f425c3)

#### Datatype:
![326149153-992ffd1f-0f1d-4431-8b28-6952814db938](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/4bc8caca-9852-45ee-8dd7-f359bb878988)

#### Labelled Data:
![326149176-ffbf35c5-b7b1-4ac6-8d48-5017d201f2eb](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/05b05f49-ac04-4929-901c-e13fa4cdb9b9)

#### Dependent variable Y:
![326149210-a34516e4-b57f-41c4-9d94-d3d43f3dce92](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/64dbdf04-cf07-4dbf-8404-ebd2ceccee00)

#### Accuracy:
![326149043-9b8ef2d9-8a67-41d2-a35c-b3c0234f0895](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/f55c9057-694d-4195-b9dd-16e56ffc2a4f)

#### Y_predicted & Y:
![326149245-63bf8995-3b82-4807-b9fc-490663a85612](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/4e0f17ff-d03f-4be2-91c4-796594806cde)

#### Y_prednew:
![326149309-6c0adeaf-b6a3-4cde-84ac-683ba144cace](https://github.com/chaitanya18c/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119392724/2fd3938e-1f7b-421b-a11c-b846fdbf8c0a)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

