import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

df = sns.load_dataset('iris')
#print(df.head())

#final_df = df[df['species'] != 'setosa']
final_df = df[df['species'].isin(['versicolor', 'virginica'])]
#print(final_df.head())

print(final_df.isnull().sum())
print(final_df['species'].unique())
final_df['species'] = final_df['species'].map({'versicolor':0,'virginica':1})
print(final_df.head())

X = final_df.iloc[:,:-1]
Y = final_df['species']

#train test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,_Y_test = train_test_split(X,Y,train_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
classifier = LogisticRegression()

parameters = {'penalty':['l1','l2','elasticnet'],'C' :[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,],'max_iter':[100,20,300]}
classifiercv = GridSearchCV(classifier,param_grid=parameters,cv=5)
classifiercv.fit(X_train,Y_train)
print(classifiercv.best_params_)
print(classifiercv.best_score_)

Y_pred = classifiercv.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
score = accuracy_score(Y_pred,_Y_test)
print(score)

print(classification_report(Y_pred,_Y_test))

sns.pairplot(final_df,hue='species')
#plt.show()

#estimated probabilites for flower with  patels  length varying 0 to 3 cm

# Create a new dataset with only the 'petal_width' feature
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)

# Repeat the 'petal_width' feature for each row in x_new
other_features_new = np.repeat(x_new, 4, axis=1)  # Repeat 'petal_width' four times

# Predict probabilities for both classes
y_proba = classifiercv.predict_proba(other_features_new)

# Plot the predicted probabilities
plt.plot(x_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(x_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.xlabel("petal_width")
plt.ylabel("Probability")
plt.legend()
plt.show()