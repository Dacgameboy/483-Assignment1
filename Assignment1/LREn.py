import time as t
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# read data from csv
df = pd.read_csv('Data1.csv')
print(df.shape)
# print(df)
# print(df.describe())


# split target and features
target_column = ['Idx']
features = list(set(list(df.columns)) - set(target_column))

# standardize the values
df[features] = df[features]/df[features].max()
# print(df.describe())


X = df[features].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=60)

# print(X_train.shape)
# print(X_test.shape)


print("\nRidge:")
rr = Ridge(alpha=0.01)

# Time for Ridge
StartRTime = t.time()
rr.fit(X_train, y_train)
EndRTime = t.time()

# Print total time when learning a model
print(f"Ridge Training Time:  {EndRTime - StartRTime}s ")

pred_train_rr = rr.predict(X_train)
print("Training RMSE: ", np.sqrt(mean_squared_error(y_train, pred_train_rr)))
print("Training R2: ", r2_score(y_train, pred_train_rr))

pred_test_rr = rr.predict(X_test)
print("Testing RMSE: ", np.sqrt(mean_squared_error(y_test, pred_test_rr)))
print("Testing R2: ", r2_score(y_test, pred_test_rr))


print("\nLasso: ")
model_lasso = Lasso(alpha=0.0001)

StartLTime = t.time()
model_lasso.fit(X_train, y_train)
EndLTime = t.time()

print(f"Lasso Training Time:  {EndLTime - StartLTime}s ")

pred_train_lasso = model_lasso.predict(X_train)
print("Training RMSE: ", np.sqrt(mean_squared_error(y_train, pred_train_lasso)))
print("Training R2: ", r2_score(y_train, pred_train_lasso))

pred_test_lasso = model_lasso.predict(X_test)
print("Testing RMSE: ", np.sqrt(mean_squared_error(y_test, pred_test_lasso)))
print("Testing R2: ", r2_score(y_test, pred_test_lasso))

# Elastic Net
print("\nElastic Net: ")
model_enet = ElasticNet(alpha=0.00001)

StartETime = t.time()
model_enet.fit(X_train, y_train)
EndETime = t.time()

print(f"Elastic Net Training Time:  {EndETime - StartETime}s ")

pred_train_enet = model_enet.predict(X_train)
print("Training RMSE: ", np.sqrt(mean_squared_error(y_train, pred_train_enet)))
print("Training R2: ", r2_score(y_train, pred_train_enet))

pred_test_enet = model_enet.predict(X_test)
print("Testing RMSE: ", np.sqrt(mean_squared_error(y_test, pred_test_enet)))
print("Testing RMSE: ", r2_score(y_test, pred_test_enet))
