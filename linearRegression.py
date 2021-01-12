import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from yellowbrick.model_selection import FeatureImportances

from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


import seaborn as sns
import warnings

warnings.simplefilter('ignore')



df = pd.read_csv('HouseSales.csv')
# pd.set_option('display.max_columns',7)
df['bedrooms'].fillna(df.mean())
df['bathrooms'].fillna(df.mean())
df.fillna(df.mean())
df['bathrooms']= df['bathrooms'].round(decimals=0)
df['bedrooms'] = df['bedrooms'].round(decimals=0)

X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','waterfront','view']]
y = df[['price']]


scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)# print(X)
X_scaled = data_scaled# scaler = preprocessing.StandardScaler().fit(X)
# data_scaled = scaler.transform(X)
scaler = preprocessing.StandardScaler().fit(y)# X_scaled = data_scaled[:,0:5]
data_scaled = scaler.transform(y)# y_scaled = data_scaled[:,6]
y_scaled = data_scaled
# print(X_scaled)

X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(X_scaled, y_scaled, test_size = 0.33, random_state = 0)
reg = LinearRegression()
reg.fit(X_full_train,Y_full_train)
# score = reg.score(X_full_train.astype(int), Y_full_test.astype(int))
y_predict = reg.predict(X_full_test)

print('r2=',r2_score(Y_full_test,y_predict))
print('MAE=',mean_absolute_error(Y_full_test,y_predict))
print('RMSE',np.sqrt(mean_squared_error(Y_full_test,y_predict)))


df = pd.DataFrame({'Actual':Y_full_test.flatten(), 'Predicted':y_predict.flatten()})
iteration = []
for i in range (len(y_predict)):
    iteration.append(i)

plt.plot(iteration[0:100], Y_full_test[0:100].flatten(),label='actual')
plt.plot(iteration[0:100], y_predict[0:100],label='predicted',color='red')
plt.legend()
plt.title('LinearRegression')
plt.show()

meanr2 = 0
meanscore = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    reg = LinearRegression()
    reg.fit(X_full_train,Y_full_train)
    # score = reg.score(X_full_train.astype(int), Y_full_test.astype(int))
    y_predict = reg.predict(X_full_test)
    # df = pd.DataFrame({'Actual':Y_full_test.astype(int), 'Predicted':y_predict.astype(int)})
    # print(df)
    r2= r2_score(Y_full_test,y_predict)
    mae=mean_absolute_error(Y_full_test,y_predict)
    rmse=np.sqrt(mean_squared_error(Y_full_test,y_predict))
    # meanscore += score
    meanr2 += r2
    meanMAE+= mae
    meanRMSE += rmse


print("meanR2 from 100 trial",meanr2/100)

print("meanMAE from 100 trial",meanMAE/100)

print("meanRMSE from 100 trial",meanRMSE/100)

viz = FeatureImportances(reg)
viz.fit(X_full_test,Y_full_test)
viz.show()


# print("meanscore from 100 trail",meanscore/100)




#
fig = plt.figure(figsize=(10,7))
sns.regplot(Y_full_test, y_predict, color='blue', marker='+')
plt.xlabel('scaled featured')
plt.ylabel('scaled label(price)')
plt.show()


