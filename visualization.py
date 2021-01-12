import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('HouseSales.csv')
df['bathrooms'] = df['bathrooms'].round(decimals=0)
df['bedrooms'] = df['bedrooms'].round(0)
# print(df['bathrooms'])

# X = df[['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']]
X = df['sqft_living'][1:10000]
Y = df['price'][1:10000]

fig = plt.figure(figsize=(10,7))

plt.scatter(X,Y)
plt.xlabel('sqft_living')
plt.ylabel('House Price')
plt.title(' sqft_living and House price Comparison')

X = df['bedrooms'][1:10000]
Y = df['price'][1:10000]
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('bedrooms')
plt.ylabel('House Price')
plt.title(' bedrooms and House price Comparison')
#
plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
# people = ('sqft_living', 'sqft_lot', 'sqft_above', 'waterfront', 'view', 'sqft_basement', 'bedrooms', 'condition','bathrooms','floors')
# y_pos = np.arange(len(people))
# performance = [100,22,12,10,8,7,6,5,4,3]
# color = ['#77AC30','#0072BD','#4DBEEE','#EDB120','#7E2F8E','#A2142F','#77AC30','#0072BD']
#
# ax.barh(y_pos, performance, align='center', color=color)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Relative Importances')
# ax.set_title('Feature Importances of 10 Features using DecisionTreeRegressor')
#
# plt.show()

people = ('sqft_living', 'view', 'waterfront', 'condition', 'bathrooms', 'floors', 'sqft_lot', 'bedrooms')
y_pos = np.arange(len(people))
performance = [100,22,21,17,15,4,-3,-18]
color = ['#77AC30','#0072BD','#4DBEEE','#EDB120','#7E2F8E','#A2142F','#77AC30','#0072BD']

ax.barh(y_pos, performance, align='center', color=color)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Relative Importances')
ax.set_title('Feature Importances of 8 Features using DecisionTreeRegressor')

plt.show()




# X = df['transmission']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Transmission')
# plt.ylabel('Car Price')
# plt.title('Car Price and Transmission Comparison')
#
# X = df['mileage']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Car Mileage')
# plt.ylabel('Car Price')
# plt.title('Car Price and Mileage Comparison')
#
# X = df['fuelType']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Fuel Type')
# plt.ylabel('Car Price')
# plt.title('Car Price and Fuel Type Comparison')
#
# X = df['tax']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Car Tax')
# plt.ylabel('Car Price')
# plt.title('Car Price and Tax Comparison')
#
# X = df['mpg']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Miles per Gallon')
# plt.ylabel('Car Price')
# plt.title('Car Price and Miles per Gallon Comparison')
#
# X = df['engineSize']
# fig = plt.figure(figsize=(10,7))
# plt.scatter(X,Y)
# plt.xlabel('Engine Size')
# plt.ylabel('Car Price')
# plt.title('Car Price and Engine Size Comparison')


plt.show()
