# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, choose input features and target price, then split into training and testing sets.
2. Create a scaled linear regression pipeline, train it on the data, and make predictions.
3. Create a polynomial (degree 2) regression pipeline with scaling, train it, and make predictions.
4. Calculate MSE, MAE, and R² for both models and plot actual vs predicted prices to compare.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: SARAN SADASIVAM
RegisterNumber:  212225040385
*/
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data=pd.read_csv("CarPrice_Assignment.csv")
data.head()
data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
data.head()
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

print("Name : SARAN SADASIVAM")
print("Reg no: 212225040385")
print("\n=== Cross Validation ===")
cv_scores = cross_val_score(model,x,y,cv=5)
print("Fold R_2 scores : ",[f"{score:.4f}" for score in cv_scores])
print(f"Average R_2     : {cv_scores.mean():.4f}")

y_pred=model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MSE : {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE : {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R_2 : {r2_score(y_test,y_pred):.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Prdicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="737" height="745" alt="image" src="https://github.com/user-attachments/assets/4e6842f1-afa4-4156-956c-c58c1d3cfa49" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
