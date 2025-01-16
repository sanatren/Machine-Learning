import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

#print(calories.head())
print(exercise.head())

exercise = pd.concat([exercise,calories.Calories],axis=1)


print(exercise.info())
print(exercise.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
exercise['Gender'] = le.fit_transform(exercise['Gender'])


exercise = exercise.astype(int)

exercise.drop(columns='User_ID',inplace=True)
print(exercise.head())

#EDA
plt.figure(figsize=(6,6))
sns.distplot(x=exercise.Body_Temp)
plt.title('Body_Temp')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(x=exercise.Heart_Rate)
plt.title('Heart_Rate')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(x=exercise.Duration)
plt.title('Duration')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(x=exercise.Age)
plt.title('Age')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(x=exercise.Gender)
plt.title('Gender')
plt.show()

plt.scatter(x=exercise['Duration'],y=exercise['Calories'],c='blue', alpha=0.5, edgecolors='w', s=100)
plt.xlabel('Duration (minutes)', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)
plt.title('duration vs calories')
plt.show()

plt.scatter(x=exercise['Heart_Rate'],y=exercise['Calories'],c='blue', alpha=0.5, edgecolors='w', s=100)
plt.xlabel('Heart_Rate', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)
plt.title('Heart_Rate vs calories')
plt.show()

plt.scatter(x=exercise['Body_Temp'],y=exercise['Calories'],c='blue', alpha=0.5, edgecolors='w', s=100)
plt.xlabel('Body_Temp', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)
plt.title('Body_Temp vs calories')
plt.show()

plt.scatter(x=exercise['Gender'],y=exercise['Calories'],c='blue', alpha=0.5, edgecolors='w', s=100)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)
plt.title('Gender vs calories')
plt.show()

plt.scatter(x=exercise['Age'],y=exercise['Calories'],c='blue', alpha=0.5, edgecolors='w', s=100)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)
plt.title('Age vs calories')
plt.show()

sns.pairplot(exercise)
plt.show()

#correletion
print(exercise.corr)
plt.figure(figsize=(36,36))
sns.heatmap(exercise.corr(),annot=True,cmap = "RdYlGn")
plt.show()

#feature selection
X = exercise.drop(columns='Calories')
Y = exercise['Calories']

print(X.columns)
print(exercise.columns)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import ExtraTreesRegressor
extra = ExtraTreesRegressor(n_jobs=-1)
print(extra.fit(X,Y))
print("feature imp: ",extra.feature_importances_)

#important features plotting
plt.figure(figsize=(12,8))
feat_importance = pd.Series(extra.feature_importances_,index = X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.show()

#model training and important feature for random forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_jobs=-1)
forest.fit(x_train,y_train)
y_pred = forest.predict(x_train)
print("predicted val:",forest.predict)

plt.figure(figsize=(12, 8))
feat_importance = pd.Series(forest.feature_importances_, index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.show()

from sklearn.metrics import r2_score
score = r2_score(y_train,y_pred)
print("random r2_score: ",score)

y_pred_test = forest.predict(x_test)
test_score = r2_score(y_test, y_pred_test)
print("Test R² score_test:", test_score)

#Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
reg_pred = regressor.predict(x_train)
score2 = r2_score(y_train,reg_pred)
print("reg score_train_regression: ",score2)

reg_pred_test = regressor.predict(x_test)
pred_test = r2_score(y_test,reg_pred_test)
print("regresion_test: ",pred_test)

#cross validation on both models for consistency
from sklearn.model_selection import cross_val_score

valid = cross_val_score(forest,X,Y,cv = 5,scoring='r2')
print("Cross-validation R² scores(randomforest):", valid)
print("Mean cross-validation R² score:", np.mean(valid))

valid = cross_val_score(regressor,X,Y,cv = 5,scoring='r2')
print("Cross-validation R² scores(regression):", valid)
print("Mean cross-validation R² score:", np.mean(valid))

#Plotting the residuals to check for patterns that might suggest problems with the model.
#calculation residuals
train_residuals = y_train - y_pred
test_residuals = y_test - y_pred_test

# Plotting residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, train_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Training Residuals')

plt.subplot(1, 2, 2)
plt.scatter(y_pred_test, test_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Test Residuals')

plt.tight_layout()
plt.show()

import pickle
# Save models using pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Save scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)