import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from yellowbrick.model_selection import FeatureImportances




import warnings

warnings.simplefilter('ignore')
df = pd.read_csv('HouseSales.csv')
pd.set_option('display.max_columns',11)
df['bedrooms'].fillna(df.mean())
df['bathrooms'].fillna(df.mean())
df.fillna(df.mean())
df['bathrooms']= df['bathrooms'].round(decimals=0)
df['bedrooms'] = df['bedrooms'].round(decimals=0)


X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','waterfront','view','sqft_above','sqft_basement']]
y = df[['price']]

scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X_scaled = data_scaled
scaler = preprocessing.StandardScaler().fit(y)
data_scaled = scaler.transform(y)
y_scaled = data_scaled


X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(X_scaled, y_scaled, test_size = 0.33, random_state = 0)
# print("X Train",len(X_full_train))
# print("X Test",len(X_full_test))
model = DecisionTreeRegressor()
model.fit(X_full_train,Y_full_train)
y_predict = model.predict(X_full_test)

# print(y_predict)
# print(Y_full_test.flatten())
print('r2=',r2_score(Y_full_test,y_predict))
print('MAE=',mean_absolute_error(Y_full_test,y_predict))
print('RMSE',np.sqrt(mean_squared_error(Y_full_test,y_predict)))

# prediciton = y_predict
# print(prediciton)
# print(df['bedrooms'].values)
#
# box = plt.boxplot(df['bedrooms'].values,widths=1, patch_artist = True )
# colors = ['cyan', 'lightgreen', 'pink']
#
# for patch, color in zip(box['boxes'], colors):
#   patch.set_facecolor(color)
#
# plt.show()

df = pd.DataFrame({'Actual':Y_full_test.flatten(), 'Predicted':y_predict})
iteration = []
for i in range (len(y_predict)):
    iteration.append(i)

plt.plot(iteration[0:100], Y_full_test[0:100].flatten(),label='actual')
plt.plot(iteration[0:100], y_predict[0:100],label='predicted',color='red')
plt.title('DecisionTreeRegressor')
plt.legend()
plt.show()

meanr2 = 0
meanscore = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    model = DecisionTreeRegressor()
    model.fit(X_full_train,Y_full_train)
    # score = reg.score(X_full_train.astype(int), Y_full_test.astype(int))
    y_predict = model.predict(X_full_test)
    # df = pd.DataFrame({'Actual':Y_full_test, 'Predicted':y_predict})
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


viz = FeatureImportances(model)
viz.fit(X_full_test,Y_full_test)
viz.show()






# xgb.plot_importance(bst, color='red')
# plt.title('importance', fontsize = 20)
# plt.yticks(fontsize = 10)
# plt.ylabel('features', fontsize = 20)


