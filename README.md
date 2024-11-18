# EX 09 : Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
### Date : 18/10/24
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
 
2. Calculate the null values present in the dataset and apply label encoder.
 
3. Determine test and training data set and apply decison tree regression in dataset.
 
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VISHAL M.A
RegisterNumber:  212222230177
*/
```
```c
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data.Head():
![ott1](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/a38d26c5-91b5-4753-bf15-f4a20f6b687b)




## Data.info():

![ott2](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/f50de7a7-93c2-4dca-9f53-5fd885edc136)



## isnull() and sum():

![ott3](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/472e96ca-3ba1-47d1-bb1d-87675a3c0233)



## Data.Head() for salary:
![ott4](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/8d408696-7586-426e-8859-9528c8dd94ca)




## MSE Value:
![ot5](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/ea307d1f-f5b9-4ce3-a84f-3cfa61ee9991)



## r2 Value:
![ot6](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/cdce43be-5386-4ac1-afd0-41d091a3e2b7)



## Data Prediction:
![ot7](https://github.com/vishal21004/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119560110/4125e73d-dda2-4ed5-bbd8-378ce0998c6f)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
