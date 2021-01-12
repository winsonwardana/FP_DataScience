import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing


from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
#
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('HouseSales.csv')
# pd.set_option('display.max_columns',7)

X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','waterfront','view','sqft_above','sqft_basement']]
y = df[['price']]


scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X_scaled = data_scaled

scaler = preprocessing.StandardScaler().fit(y)
data_scaled = scaler.transform(y)
y_scaled = data_scaled
# print(X_scaled)

X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(X_scaled, y_scaled, test_size = 0.33, random_state = 0)
model = neighbors.KNeighborsRegressor(n_neighbors=7)
model.fit(X_full_train, Y_full_train)
y_predict = model.predict(X_full_test)
df = pd.DataFrame({'Actual':Y_full_test.flatten(), 'Predicted':y_predict.flatten()})
print(len(y_predict))
iteration = []
for i in range (len(y_predict)):
    iteration.append(i)

plt.plot(iteration[0:100], Y_full_test[0:100],label='actual')
plt.plot(iteration[0:100], y_predict[0:100],label='predicted',color='red')
plt.legend()
plt.title('K-Nearest Neighbor')
plt.show()

rmse_val = [] #to store rmse values for different k

for K in range(20):
    K +=1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)
    model.fit(X_full_train, Y_full_train)
    pred = model.predict(X_full_test)
    # df = pd.DataFrame({'Actual':Y_full_test, 'Predicted':pred.astype(int)})
    # print(df)

    error = sqrt(mean_squared_error(Y_full_test, pred)) #calculate rmse
    rmse_val.append(error)
    print('RMSE value for k=', K, ' is: ', error)
    r2= r2_score(Y_full_test,pred)
    mae=mean_absolute_error(Y_full_test,pred)
    print('r2 score is', r2)
    print('mae score is', mae)

meanr2 = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    model = neighbors.KNeighborsRegressor(n_neighbors=7)
    model.fit(X_full_train, Y_full_train)
    pred = model.predict(X_full_test)
    # df = pd.DataFrame({'Actual':Y_full_test, 'Predicted':pred.astype(int)})
    # print(df)

    error = sqrt(mean_squared_error(Y_full_test, pred)) #calculate rmse
    rmse_val.append(error)

    r2= r2_score(Y_full_test,pred)
    mae=mean_absolute_error(Y_full_test,pred)

    meanr2 += r2
    meanMAE += mae
    meanRMSE += error

print("meanR2 from 100 trial",meanr2/100)

print("meanMAE from 100 trial",meanMAE/100)

print("meanRMSE from 100 trial",meanRMSE/100)


# print(rmse_val)
# # print(re)
# # print('printing elbow curve')
# n = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# fig = plt.figure(figsize=(10,7))
# plt.bar(n,rmse_val)
# plt.title('K neighbors with its Mean Squared Error value')
# plt.xlabel('n K neighbor')
# plt.ylabel('Mean Squared Error')
# plt.show()








