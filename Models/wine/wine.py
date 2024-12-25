import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None) 

red_df = pd.read_csv('winequality-red.csv')
print(red_df.head())
#print(red_df.isnull().sum())
#print(red_df.info())

#sns.pairplot(red_df)
#plt.show()

X = red_df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
Y = red_df['quality']

fig, axs = plt.subplots(3, 4, figsize=(15, 10))

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Scatter plot for each feature
for i, (column, ax) in enumerate(zip(X.columns, axs)):
    ax.scatter(X[column], Y)
    ax.set_xlabel(column)
    ax.set_ylabel('Quality')

# Adjust layout
plt.tight_layout()
#plt.show()

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

#standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train,Y_train)
print("prediction",regressor.predict)
print("coffiencient",regressor.coef_)
print("intercept",regressor.intercept_)

from sklearn.model_selection import cross_val_score
valid_score = cross_val_score(regressor,X_train,Y_train,scoring='neg_mean_squared_error',cv = 5)
np.mean(valid_score)

y_pred = regressor.predict(X_test)
print("ypred = ",y_pred)



from sklearn.metrics import r2_score ,mean_absolute_error , mean_squared_error
mae = mean_absolute_error(Y_test,y_pred)
mse = mean_squared_error(Y_test,y_pred)
rsme = np.sqrt(mse)
score = r2_score(Y_test,y_pred)
print("r2 is ",score)

#adjusted r^2
print(1 - (1-score)*(len(Y_test)-1) /(len(Y_test)-X_test.shape[1]-1))

#ridge regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge_regress = Ridge()

parameter = {'alpha':[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]}
ridgecv = GridSearchCV(ridge_regress,param_grid=parameter,scoring = 'neg_mean_squared_error',cv = 5)
ridgecv.fit(X_train,Y_train)
ridge_pred = ridgecv.predict(X_test)

print("best parameter is ",ridgecv.best_params_)
print("ridge prediction is ",ridge_pred)

#r2 score after applying ridge regresssion
score1 = r2_score(Y_test,ridge_pred)
print("r2 (ridge) ",score1)

#lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso()

parameter = {'alpha':[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]}
lassocv = GridSearchCV(lasso,param_grid=parameter,scoring = 'neg_mean_squared_error',cv = 5)
lassocv.fit(X_train,Y_train)
lasso_pred = lassocv.predict(X_test)

print("best para ",lassocv.best_params_)
print("score ",lassocv.best_score_)
lassocv.fit(X_train,Y_train)
lass_pred = lassocv.predict(X_test)
print("lasso pred is " ,lass_pred)

#r2 score of lasso is
score2 = r2_score(Y_test,lass_pred)
print("r2 (lasso) ",score2)

from  sklearn.neighbors import KNeighborsClassifier

KNeighbor = KNeighborsClassifier()
KNeighbor.fit(X_train,Y_train)

neighbor_pred = KNeighbor.predict(X_test)
#assumptions

plt.scatter(Y_test,y_pred)
plt.show()

residuals = Y_test - y_pred
sns.displot(residuals,kind='kde')
plt.show()

sns.displot(ridge_pred-Y_test,kind='kde')
plt.show()

sns.displot(lass_pred-Y_test,kind='kde')
plt.show()


plt.scatter(y_pred,residuals)
plt.show()

from statsmodels.api import OLS
model = OLS(Y_train,X_train).fit()
prediction = model.predict(X_test)
print(prediction)
print(model.summary())

lassocv.fit(X_train, Y_train)
print(lassocv.predict(scaler.transform([[11.2,0.28,0.50,1.9,0.075,17.0,64.0,0.9980,3.16,0.58,16.8]])))