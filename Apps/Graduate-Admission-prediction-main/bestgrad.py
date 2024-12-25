import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet, HuberRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load the dataset
pd.set_option('display.max_columns', None)
Admission = pd.read_csv('Admission_Predict_Ver1.1.csv')

# Initial inspection
print(Admission.head())
print(Admission.isnull().sum())
print(Admission.info())
print(Admission.duplicated().sum())

# Drop 'Serial No.' column
Admission = Admission.drop(columns='Serial No.')

# Check column names after dropping 'Serial No.'
print("Column names after dropping 'Serial No.':", Admission.columns)

# Rename columns
Admission = Admission.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'})

# Comparing features
sns.regplot(x="GRE Score", y="TOEFL Score", data=Admission)
plt.title("GRE Score vs TOEFL Score")
plt.show()

sns.regplot(x="GRE Score", y="CGPA", data=Admission)
plt.title("GRE Score vs CGPA")
plt.show()

sns.regplot(x="LOR", y="CGPA", data=Admission)
plt.title("LOR vs CGPA")
plt.show()

sns.lmplot(x="GRE Score", y="LOR", data=Admission, hue="Research")
plt.title("GRE Score vs LOR")
plt.show()

sns.lmplot(x="CGPA", y="LOR", data=Admission, hue="Research")
plt.title("CGPA vs LOR")
plt.show()

sns.regplot(x="CGPA", y="SOP", data=Admission)
plt.title("CGPA vs SOP")
plt.show()

sns.regplot(x="GRE Score", y="SOP", data=Admission)
plt.title("GRE Score vs SOP")
plt.show()

sns.regplot(x="TOEFL Score", y="SOP", data=Admission)
plt.title("TOEFL Score vs SOP")
plt.show()

#correlation
corr = Admission.corr(numeric_only=True)
#heatmap of correlation
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,linewidths=1.5,cmap="YlGnBu", annot_kws={"size": 8})


# Split the data
X = Admission.drop(columns='Chance of Admit')
Y = Admission['Chance of Admit']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Standardize the features
minmax = MinMaxScaler()
x_train_scaled = minmax.fit_transform(x_train)
x_test_scaled = minmax.transform(x_test)

# List of models to evaluate
models = [
    ['DecisionTree:', DecisionTreeRegressor()],
    ['Linear Regression:', LinearRegression()],
    ['RandomForest:', RandomForestRegressor()],
    ['KNeighbours:', KNeighborsRegressor(n_neighbors=2)],
    ['SVM:', SVR()],
    ['AdaBoost:', AdaBoostRegressor()],
    ['GradientBoosting:', GradientBoostingRegressor()],
    ['Xgboost:', XGBRegressor()],
    ['CatBoost:', CatBoostRegressor(logging_level='Silent')],
    ['Lasso:', Lasso()],
    ['Ridge:', Ridge()],
    ['BayesianRidge:', BayesianRidge()],
    ['ElasticNet:', ElasticNet()],
    ['HuberRegressor:', HuberRegressor()]
]

# Evaluate each model
print("Results...")

for name, model in models:
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"{name} RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

param_grid_ridge = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}
# extratreetRegressor
from sklearn.tree import ExtraTreeRegressor
extra = ExtraTreeRegressor()
extra.fit(x_train,y_train)

# Feature importance from extratreetRegressor
plt.figure(figsize=(12,8))
feat_importance = pd.Series(extra.feature_importances_,index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.show()

#picking best models to  train
#Ridge regression
grid_search_ridge = GridSearchCV(estimator=Ridge(), param_grid=param_grid_ridge, cv=5, n_jobs=-1, verbose=2)
grid_search_ridge.fit(x_train_scaled, y_train)

best_ridge = grid_search_ridge.best_estimator_
y_pred_best_ridge = best_ridge.predict(x_test_scaled)
score_best_ridge = r2_score(y_test, y_pred_best_ridge)
print("Best Ridge R2 Score: ", score_best_ridge)

# RandomForestRegressor with Grid Search CV
random = RandomForestRegressor(random_state=42)
random.fit(x_train_scaled, y_train)
y_pred_rand = random.predict(x_test_scaled)
score_rand = r2_score(y_test, y_pred_rand)
print("R2 score for random forest: ", score_rand)

# Feature importance from RandomForestRegressor
plt.figure(figsize=(12, 8))
feat_imp = pd.Series(random.feature_importances_, index=X.columns)
feat_imp.nlargest(20).plot(kind='barh')
plt.title("Feature Importances (Random Forest)")
plt.show()


param_grid_forest = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_forest = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_forest, cv=5, n_jobs=-1, verbose=2)
grid_search_forest.fit(x_train_scaled, y_train)
grid_search_forest.fit(x_train_scaled, y_train)

best_forest = grid_search_forest.best_estimator_
y_pred_best_forest = best_forest.predict(x_test_scaled)
score_best_forest = r2_score(y_test, y_pred_best_forest)
print("Best Random Forest R2 Score: ", score_best_forest)

# Linear Regression
reg = LinearRegression()
reg.fit(x_train_scaled, y_train)
y_pred_reg = reg.predict(x_test_scaled)
score_reg = r2_score(y_test, y_pred_reg)
print("Linear Regression R2 Score:", score_reg)

# Feature importance for Linear Regression
coef = reg.coef_
feature_importance = np.abs(coef)

plt.figure(figsize=(12, 8))
feat_importance = pd.Series(feature_importance, index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.title("Feature Importances (Linear Regression)")
plt.show()

#since LinearRegression is performing best so assumption to verify model 

plt.scatter(y_test,y_pred_reg)
plt.show()

#residuals calculation
residuals = y_test - y_pred_reg
sns.displot(residuals,kind='kde')
plt.show()

#scatter plot with respect to residuals and predictions
plt.scatter(y_pred_reg,residuals)
plt.show()

#importing OLS for model summary
import statsmodels.api as sm
model = sm.OLS(y_train,x_train_scaled).fit()
prediction = model.predict(x_test)
print(prediction)
print(model.summary())

#saving the file for streamlit deployment
import pickle

# Load the model and scaler
# Save the model
model_filename = 'bestgrad.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(reg, model_file)

# Save the scaler
scaler_filename = 'minmax.pkl'
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(minmax, scaler_file)