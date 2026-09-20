# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
csv_path = "Salary.csv"    # <-- Change path if needed
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

# 2) Data exploration
print("Shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# 3) Encode categorical column 'Position'
if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 4) Select features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
display(X.head())
print("\nTarget Sample:")
display(y.head())

# 5) Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

# 6) Initialize and train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

# 7) Predict on test data
y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

# 8) Evaluate model using R² Score
r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

# 9) Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# 10) Feature Importances
importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
display(importances)

```
## Output:

<img width="1750" height="651" alt="image" src="https://github.com/user-attachments/assets/28e38e15-84ff-4d0a-ae8b-1276bc490cf8" />
<img width="598" height="492" alt="image" src="https://github.com/user-attachments/assets/febf98d5-69e4-4c60-9a26-316f59df62fe" />
<img width="1100" height="775" alt="image" src="https://github.com/user-attachments/assets/0ba6ec30-0e57-4f53-b51a-4b18b2c02d7c" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
