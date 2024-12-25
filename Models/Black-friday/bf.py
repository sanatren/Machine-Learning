import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = pd.concat([df_train,df_test])#merging datasets
df.drop('User_ID',axis = 1,inplace = True)#delete feilds by rows if axis =0 and col by axis = 1 ,inplace to update df
#print(df.head())

#handling categorical values 
df['Gender'] = df['Gender'].map({'M':0,'F':1})

print((df['Age'].unique()))
df['Age'] = df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
df_city = pd.get_dummies(df['City_Category'] ,drop_first= True, dtype=int)
df = pd.concat([df,df_city],axis=1)#axis = 1 cuz we need to show all vals
df.drop(['City_Category'],axis=1,inplace=True)
#print(df.head())

#handling missing values

modee = df['Product_Category_2'].mode()[0]
df['Product_Category_2'] = df['Product_Category_2'].fillna(modee)


mode2 = df['Product_Category_3'].mode()[0]
df['Product_Category_3'] = df['Product_Category_3'].fillna(mode2)#replacing nan values with mode of this datafeild
print(df.isnull().sum())
#print(df['Product_Category_3'].unique())


#handling staying in current city years


df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+','')
print(df.head())



#converting object to int
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)
#df['B'] = df['B'].astype(int)
#df['C'] = df['C'].astype(int)

print(df.info())
#sns.pairplot(df)
#sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)#age vs purchase
#sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)
sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)
plt.show()

##Feature Scaling 
df_test=df[df['Purchase'].isnull()]
df_train=df[~df['Purchase'].isnull()]
X=df_train.drop('Purchase',axis=1)
X.head()
y=df_train['Purchase']
X.shape
y=df_train['Purchase']
y.shape

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)
X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(X_test)