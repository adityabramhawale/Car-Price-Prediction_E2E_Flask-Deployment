#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset
c=pd.read_csv("C:\\Users\\bramhad\\OneDrive - Konecranes Plc\\Desktop\\Data Science\\End to End Projects\\Car\\car data.csv")
print(c.shape)

c1=c[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
c1.head

c1['Present_Year']=2021
c1['Number_of_Years_Old']=c1['Present_Year']- c1['Year']
c1.head()

c1.drop(labels=['Year', 'Present_Year'],axis=1,inplace=True)
c1.head()


#select categorical variables from then dataset, and then implement categorical encoding for nominal variables
Fuel_Type=c1[['Fuel_Type']]
Fuel_Type=pd.get_dummies(Fuel_Type, drop_first=True)

Seller_Type=c1[['Seller_Type']]
Seller_Type=pd.get_dummies(Seller_Type, drop_first=True)

Transmission=c1[['Transmission']]
Transmission=pd.get_dummies(Transmission, drop_first=True)

c2=pd.concat([c1,Fuel_Type, Seller_Type, Transmission], axis=1)

c2.drop(labels=['Fuel_Type', 'Seller_Type', 'Transmission'], axis=1, inplace=True)

c2.head()

sell=c2['Selling_Price']
c2.drop(['Selling_Price'], axis=1, inplace=True)
c3=c2.join(sell)
c3.head()

x=c3.iloc[:,:-1]
y=c3.iloc[:,-1]

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state = 0)
dt_reg.fit(x_train, y_train)
y_pred=dt_reg.predict(x_test)

print("Decision Tree Score on Training set is",dt_reg.score(x_train, y_train))#Training Accuracy
print("Decision Tree Score on Test Set is",dt_reg.score(x_test, y_test))#Testing Accuracy

accuracies = cross_val_score(dt_reg, x_train, y_train, cv = 5)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

mae=mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:" , mae)

mse=mean_squared_error(y_test, y_pred)
print("Mean Squared Error:" , mse)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('The r2_score is', metrics.r2_score(y_test, y_pred))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=400,min_samples_split=15,min_samples_leaf=2,
max_features='auto', max_depth=30)
rf_reg.fit(x_train, y_train)
y_pred=rf_reg.predict(x_test)

print("Random Forest Score on Training set is",rf_reg.score(x_train, y_train))#Training Accuracy
print("Random Forest Score on Test Set is",rf_reg.score(x_test, y_test))#Testing Accuracy

accuracies = cross_val_score(rf_reg, x_train, y_train, cv = 5)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

mae=mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:" , mae)

mse=mean_squared_error(y_test, y_pred)
print("Mean Squared Error:" , mse)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('The r2_score is', metrics.r2_score(y_test, y_pred))

from sklearn.ensemble import VotingRegressor
vot_reg = VotingRegressor([('DecisionTree', dt_reg), ('RandomForestRegressor', rf_reg)])
vot_reg.fit(x_train, y_train)
y_pred=vot_reg.predict(x_test)

print("Voting Regresssor Score on Training set is",vot_reg.score(x_train, y_train))#Training Accuracy
print("Voting Regresssor Score on Test Set is",vot_reg.score(x_test, y_test))#Testing Accuracy

accuracies = cross_val_score(vot_reg, x_train, y_train, cv = 5)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

mae=mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:" , mae)

mse=mean_squared_error(y_test, y_pred)
print("Mean Squared Error:" , mse)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('The r2_score is', metrics.r2_score(y_test, y_pred))

import pickle
pickle.dump(vot_reg, open("vot_reg.pkl", "wb"))

# load model from file
model = pickle.load(open("vot_reg.pkl", "rb"))

model.predict([[9.85, 6900, 0, 3, 1, 0, 1, 0]])



