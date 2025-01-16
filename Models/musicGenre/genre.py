import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('songs.csv')

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None) 


print(df.info())
#print(df.isnull().sum())

df.fillna(method='ffill' , inplace = True)

# Replace missing values in categorical columns
categorical_columns = ['composer', 'publisher', 'lyricist']
for column in categorical_columns:
    df[column].fillna('Unknown', inplace=True)

numerical_columns = ['bit_rate', 'comments', 'duration', 'favorites', 'listens']
for column in numerical_columns:
    df[column].fillna(df[column].mean(), inplace=True)



df['date_created'] = pd.to_datetime(df['date_created'])
df['year_created'] = df['date_created'].dt.year
df['month_created'] = df['date_created'].dt.month
df['day_created'] = df['date_created'].dt.day

df.drop(columns = ['date_created','track_id','date_recorded','genres','genres_all','information','license','tags','language_code'],inplace=True)



#encdeing
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['composer'] = label.fit_transform(df['composer'])
df['lyricist'] = label.fit_transform(df['lyricist'])
df['publisher'] = label.fit_transform(df['publisher'])
df['genre_top'] = label.fit_transform(df['genre_top'])

df['composer'] = df['composer'].astype(int)
df['lyricist'] = df['lyricist'].astype(int)
df['publisher'] = df['publisher'].astype(int)
df['genre_top'] = df['genre_top'].astype(int)

title = df[['title']]
title = pd.get_dummies(title,drop_first=True)
title.astype(int)

df = pd.concat([df,title],axis=1)
df.drop(columns = 'title',axis =1 , inplace = True)

print(df.isnull().sum())

#defing features and targets
X = df.drop(columns='genre_top')
Y = df['genre_top']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
x_train = SS.fit_transform(x_train)
x_test = SS.transform(x_test)

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
print(selection.fit(X,Y))
print("feature imp: ",selection.feature_importances_)

plt.figure(figsize=(20,20))
feat_importance = pd.Series(selection.feature_importances_,index = X.columns)
feat_importance.nlargest(20).plot(kind = 'barh')
plt.show()

from sklearn.tree import DecisionTreeRegressor
trie = DecisionTreeRegressor(random_state=10)
trie.fit(x_train,y_train)
trie_pred = trie.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(random_state=10)
reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

from sklearn.metrics import r2_score
tree_rep = r2_score(y_test, trie_pred)
reg_rep = r2_score(y_test, reg_pred)
print("tree_regression_accuracy: ",tree_rep)
print("random_forest_accuracy: ",reg_rep)



