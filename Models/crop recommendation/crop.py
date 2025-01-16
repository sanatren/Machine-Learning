import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('crop.csv')

print(df['label'].unique())

crop_mapping = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5, 'mothbeans': 6, 
    'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12, 
    'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16, 'orange': 17, 'papaya': 18, 
    'coconut': 19, 'cotton': 20, 'jute': 21, 'coffee': 22
}
df['labeling'] = df['label'].map(crop_mapping)
print(df.tail())

print(df.isnull().sum())
print(df.info())

X = df[['N','P','K','temperature','humidity','ph','rainfall']]
Y = df['labeling']

#train split

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train,Y_train)
print("coff ",regressor.coef_)
print("inter ",regressor.intercept_)
print("predicted ",regressor.predict)

from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,X_train,Y_train,scoring='neg_mean_squared_error',cv = 5)
print("cross validity  :",np.mean(score))
 
Y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
mae = mean_absolute_error(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(mse)
score1 = r2_score(Y_test,Y_pred)
print("mae is: ",mae)
print("mse is: ",mse)
print("rmse is: ",rmse)
print("r2 score : ",score1)

from sklearn.ensemble import RandomForestRegressor
ram = RandomForestRegressor(max_depth=100,min_samples_split=5,min_samples_leaf=2,max_features= 'log2')
ram.fit(X_train,Y_train)

Z_pred = ram.predict(X_test)

from sklearn.metrics import r2_score
r2_sc1=r2_score(Y_test,Z_pred)
print("r2 sc",r2_sc1)