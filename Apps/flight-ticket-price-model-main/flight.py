
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
from sklearn.model_selection import  GridSearchCV
from scipy.stats import mode
df_train = pd.read_excel('dataset/Data_Train.xlsx')

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None) 
print(df_train.head())


print(df_train.isnull().sum())

df_train['Date'] = df_train['Date_of_Journey'].str.split('/').str[0]
df_train['Month'] = df_train['Date_of_Journey'].str.split('/').str[1]
df_train['year'] = df_train['Date_of_Journey'].str.split('/').str[2]

df_train['Date'] = df_train['Date'].astype(int)
df_train['Month'] = df_train['Month'].astype(int)
df_train['year'] = df_train['year'].astype(int)

df_train['year'].isnull().sum()
df_train['Month'].isnull().sum()
df_train['Date'].isnull().sum()

df_train.drop(columns = 'Date_of_Journey' , inplace = True,axis = 1)

df_train['Dep_Time_hrs'] = df_train['Dep_Time'].str.split(':').str[0]
df_train['Dep_Time_Mns'] = df_train['Dep_Time'].str.split(':').str[1]

df_train['Dep_Time_hrs'] = df_train['Dep_Time_hrs'].astype(int)
df_train['Dep_Time_Mns'] = df_train['Dep_Time_Mns'].astype(int)

df_train['Dep_Time_hrs'].isnull().sum()
df_train['Dep_Time_Mns'].isnull().sum()

df_train.drop(columns = 'Dep_Time',inplace = True,axis = 1)

df_train['Arrival_Time_hours'] = df_train['Arrival_Time'].str.split(':').str[0]
df_train['Arrival_Time_Minutes'] = df_train['Arrival_Time'].str.split(':').str[1]
df_train['Arrival_Time_Minutes'] = df_train['Arrival_Time_Minutes'].str.split(' ').str[0]

df_train['Arrival_Time_hours'] = df_train['Arrival_Time_hours'].astype(int)
df_train['Arrival_Time_Minutes'] = df_train['Arrival_Time_Minutes'].astype(int)

df_train['Arrival_Time_hours'].isnull().sum()
df_train.drop(columns = 'Arrival_Time',inplace = True,axis = 1)

df_train['Duration_hrs'] = df_train['Duration'].str.split(' ').str[0].str.split('h').str[0]
df_train['Duration_Mins'] = df_train['Duration'].str.split(' ').str[1].str.split('m').str[0]

duration_split = df_train['Duration'].str.extract(r'(?:(\d+)h)? ?(?:(\d+)m)?')
df_train['Duration_hrs'] = duration_split[0].fillna(0).astype(int)
df_train['Duration_Mins'] = duration_split[1].fillna(0).astype(int)

print(df_train['Duration_hrs'].isnull().sum())
print(df_train['Duration_Mins'].isnull().sum())
df_train.drop(columns='Duration', inplace=True)

df_train['Total_Stops'] = df_train['Total_Stops'].map({'non-stop': 0,'1 stop': 1,'2 stop': 2,'3 stop' : 3,'4 stop': 4})
print(df_train['Total_Stops'].isnull().sum())
print(mode(df_train['Total_Stops']))
df_train['Total_Stops']=df_train['Total_Stops'].fillna(1)
df_train['Total_Stops'] = df_train['Total_Stops'].astype(int)

df_train.drop(columns='Route',axis =1, inplace=True)
print(df_train.head())

mode = mode(df_train['Price'])
print(mode)
print(df_train['Price'].isnull().sum())
df_train['Price'].fillna(df_train['Price'].median(), inplace=True)

df_train.drop(columns = 'Additional_Info',inplace = True)

#encoding
print(df_train["Airline"].value_counts())
# Airline vs Price
sns.catplot(y = "Price", x = "Airline", data = df_train.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


Airline = df_train[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)
Airline = Airline.astype(int)

# source vs price 
sns.catplot(y = "Price", x = "Source", data = df_train.sort_values("Price",ascending = False),kind= "boxen",height=6,aspect = 3)
plt.show()

#Source onehot encoding
Source = df_train[["Source"]]
Source = pd.get_dummies(Source,drop_first=True)
Source = Source.astype(int)

#Destination
Destination = df_train[["Destination"]]
Destination = pd.get_dummies(Destination,drop_first=True)
Destination = Destination.astype(int)

df_train = pd.concat([df_train, Airline, Source, Destination], axis = 1)
df_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
print(df_train.head())

#Ed for test dataset

test_data = pd.read_excel('dataset/Test_set.xlsx')
print(test_data.head())

print(test_data.isnull().sum())


test_data['Date'] = test_data['Date_of_Journey'].str.split('/').str[0]
test_data['Month'] = test_data['Date_of_Journey'].str.split('/').str[1]
test_data['Year'] = test_data['Date_of_Journey'].str.split('/').str[2]


test_data['Date'] = test_data['Date'].astype(int)
test_data['Month'] = test_data['Month'].astype(int)
test_data['Year'] = test_data['Year'].astype(int)


test_data.drop(columns='Date_of_Journey', inplace=True)


test_data['Dep_Time_hrs'] = test_data['Dep_Time'].str.split(':').str[0]
test_data['Dep_Time_Mns'] = test_data['Dep_Time'].str.split(':').str[1]


test_data['Dep_Time_hrs'] = test_data['Dep_Time_hrs'].astype(int)
test_data['Dep_Time_Mns'] = test_data['Dep_Time_Mns'].astype(int)


test_data.drop(columns='Dep_Time', inplace=True)


test_data['Arrival_Time_hours'] = test_data['Arrival_Time'].str.split(':').str[0]
test_data['Arrival_Time_Minutes'] = test_data['Arrival_Time'].str.split(':').str[1]
test_data['Arrival_Time_Minutes'] = test_data['Arrival_Time_Minutes'].str.split(' ').str[0]


test_data['Arrival_Time_hours'] = test_data['Arrival_Time_hours'].astype(int)
test_data['Arrival_Time_Minutes'] = test_data['Arrival_Time_Minutes'].astype(int)


test_data.drop(columns='Arrival_Time', inplace=True)


duration_split = test_data['Duration'].str.extract(r'(?:(\d+)h)? ?(?:(\d+)m)?')
test_data['Duration_hrs'] = duration_split[0].fillna(0).astype(int)
test_data['Duration_Mins'] = duration_split[1].fillna(0).astype(int)


test_data.drop(columns='Duration', inplace=True)


test_data['Total_Stops'] = test_data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stop': 2, '3 stop': 3, '4 stop': 4})


test_data['Total_Stops'].fillna(1, inplace=True)
test_data['Total_Stops'] = test_data['Total_Stops'].astype(int)


test_data.drop(columns='Route', axis=1, inplace=True)


test_data.drop(columns='Additional_Info', inplace=True)


Airline = pd.get_dummies(test_data[['Airline']], drop_first=True)
Airline = Airline.astype(int)

Source = pd.get_dummies(test_data[['Source']], drop_first=True)
Source = Source.astype(int)

Destination = pd.get_dummies(test_data[['Destination']], drop_first=True)
Destination= Destination.astype(int)

test_data = pd.concat([test_data, Airline, Source, Destination], axis=1)


test_data.drop(columns=['Airline', 'Source', 'Destination'], axis=1, inplace=True)


print(test_data.head())

print(test_data.isnull().sum())

#Feature Selection
#heatmap
#feature_importance_
#SelectKBest

print(df_train.shape)
print(df_train.columns)

X = df_train.loc[:,['Total_Stops', 'Date', 'Month', 'year', 'Dep_Time_hrs',
       'Dep_Time_Mns', 'Arrival_Time_hours', 'Arrival_Time_Minutes',
       'Duration_hrs', 'Duration_Mins', 'Airline_Air India', 'Airline_GoAir',
       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
print(X.head())

Y = df_train.iloc[:,1]
print(Y.head())

#correlation
plt.figure(figsize=(36,36))
sns.heatmap(df_train.corr(),annot=True,cmap = "RdYlGn")
plt.show()

#finding important features using extra regressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
print(selection.fit(X,Y))

print(selection.feature_importances_)

#graph of feature importance
plt.figure(figsize=(12,8))
feat_importance = pd.Series(selection.feature_importances_,index=X.columns)
feat_importance.nlargest(20).plot(kind = 'barh')
plt.show()

#model training
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()
reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)
print(reg.score(X_train, Y_train))
print(reg.score(X_test, Y_test))

sns.distplot(Y_test-Y_pred)
plt.show()

plt.scatter(Y_test, Y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE:', mean_absolute_error(Y_test, Y_pred))
print('MSE:', mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(mean_squared_error(Y_test, Y_pred)))

# RMSE/(max(DV)-min(DV))

print(2090.5509/(max(Y)-min(Y)))

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100 , stop = 1200,num = 12)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num = 6)]
min_samples_split = [2,5,10,15,100]
min_sample_leaf = [1,2,5,10]

random_grid = {'n_estimators' :n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_sample_leaf
               }

random_rf = RandomizedSearchCV(estimator= reg, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv = 5,verbose = 2,random_state=42,n_jobs=-1)
random_rf.fit(X_train,Y_train)

print("best para: ", random_rf.best_params_)

prediction = random_rf.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(Y_test-prediction)
plt.show()

plt.figure(figsize = (8,8))
plt.scatter(Y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

print("errors for prediction")
print('MAE:', mean_absolute_error(Y_test, prediction))
print('MSE:', mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(mean_squared_error(Y_test, prediction)))

import pickle
#saving file
file = 'finalModel.pkl'
pickle.dump(random_rf,open(file,'wb'))

#loading the save model
loaded_model = pickle.load(open('finalModel.pkl','rb'))

y_prediction = loaded_model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_test,y_prediction))
