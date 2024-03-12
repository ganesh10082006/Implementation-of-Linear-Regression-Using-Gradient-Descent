# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ganesh G.
RegisterNumber: 212223230059 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:

#### Profit prediction:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/05887c6b-e37a-4a6b-97f2-da31cec41657)

#### Function output:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/ab6d80d1-2c57-4303-bce4-790d90ca992b)

#### Gradient descent:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/2fcbf55b-78eb-4b41-9b3a-c9497c71553b)

#### Cost function using Gradient Descent:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/8abb45a0-e650-43a0-b62b-6add3135d91b)

#### Linear Regression using Profit Prediction:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/e2f1d801-aca5-472c-a84f-3e52d679adc8)

#### Profit Prediction for a population of 35000:
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/b6100ff4-bec7-4fee-88f7-bdcb61861cc7)

#### Profit Prediction for a population of 70000 :
![image](https://github.com/ganesh10082006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151981672/4b8f41e5-f74a-4627-a421-6b3f6468099e)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
