import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('housing.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())

sns.pairplot(df)
#plt.show()

corr_matrix = df.corr()
#relevant_features = corr_matrix['MEDV'].sort_values(ascending=False).index[:3]  # Select top correlated features
X = df[['RM','LSTAT','PTRATIO']]
Y = df['MEDV']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(x_train,y_train)



from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,X,Y,scoring='neg_mean_squared_error',cv = 3)
print(np.mean(score))

Y_pred_linear = regressor.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y_test,Y_pred_linear)
mse = mean_squared_error(y_test,Y_pred_linear)
rmse = np.sqrt(mse)
print("mse is :",mse)
print("mae is :",mae)
print("rmse is :",rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,Y_pred_linear)
print("r2 of linear reg :",r2)

#adjusted r^2
print("adjusted r2: ",1 - (1-score)*(len(y_test)-1) /(len(y_test)-x_test.shape[1]-1))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
parameter = {'alpha':[0.1,200]}
ridgecv = GridSearchCV(ridge,param_grid=parameter,scoring='',cv=5)
ridge.fit(x_train,y_train)
ridge_pred = ridge.predict(x_test)

score1 = r2_score(y_test,ridge_pred)
print("r2 (ridge) ",score1)

from sklearn.linear_model import Lasso
lasso = Lasso()

parameter1 = {'alpha':[0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]}
lassocv = GridSearchCV(lasso, param_grid=parameter1, scoring='neg_mean_squared_error', cv=5)
lassocv.fit(x_train, y_train)
lasso_pred = lassocv.predict(x_test)

print("Best parameters for Lasso:", lassocv.best_params_)
print("Best score for Lasso:", lassocv.best_score_)

# Evaluate using mean squared error
mse_lasso = mean_squared_error(y_test, lasso_pred)
print("Mean Squared Error (Lasso):", mse_lasso)

# You can also evaluate using mean absolute error or other appropriate metrics
mae_lasso = mean_absolute_error(y_test, lasso_pred)
print("Mean Absolute Error (Lasso):", mae_lasso)

# Alternatively, if you still want to use R2 score for evaluation
r2_lasso = r2_score(y_test, lasso_pred)
print("R2 Score (Lasso):", r2_lasso)






