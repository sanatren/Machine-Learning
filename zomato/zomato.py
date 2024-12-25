##EDA
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('zomato.csv',encoding='latin-1')

print(df.columns )
print(df.info()) ##types of colums
print(df.describe())##5 number summary

##IN DATA ANALYSIS WHAT ALL THINGS WE DO
# MISSING VALUES
# EXPLORE ABOUT THE NUMERICAL VALUES
# EXPLORE ABOUT CATEGORICAL VARIABLE
# FINDING RELATIONSSHIP BETWEEN FEATURES

print(df.isnull().sum())#finding no of null values in dataset feilds

null_vals_holder = [features for features in df.columns if df[features].isnull().sum()>0]
print(null_vals_holder)

sns.heatmap(df.isnull(),xticklabels = False , cbar = True,cmap= 'viridis')
plt.show()

df_country = pd.read_excel('Country-Code.xlsx')
holder =pd.merge(df,df_country,on='Country Code',how='left')#function used to combine two fields of different datasets
#print(holder.head()) 

print(holder.Country.value_counts())
country_names = holder.Country.value_counts().index#storing names of country  
country_values = holder.Country.value_counts().values #creating arrays of number of values

##pie chart

plt.pie(country_values,labels = country_names)
#top three countries that use zomato

plt.pie(country_values[:3],labels = country_names[:3])

#least three countries
##plt.pie(country_values[5:8],labels=country_names[5:8])
plt.pie(country_values[12:15],labels=country_names[12:15],autopct ='%1.2f%%')
plt.show()

#Observation:Zomato minimum records or transaction from canada after that sri lanka then qatar

#group = holder.groupby(['Aggregate rating','Country Code','Rating color','Rating text']).size()
#print(group)
rating = holder.groupby(['Aggregate rating', 'Country Code', 'Rating color', 'Rating text']).size().reset_index(name='Rating_count')
plt.rcParams['figure.figsize'] = (18,12)
sns.barplot(x="Aggregate rating",y = "Rating_count",hue = 'Rating color',data = rating)
sns.barplot(x="Aggregate rating",y = "Rating_count",hue = 'Rating color',data = rating,palette = ['silver','red','orange','yellow','green','green'])
sns.countplot(x="Rating color",data = rating,palette = ['silver','red','orange','yellow','green','green'])
plt.show()

#find the names of countrie that given 0 rating
#print(holder.columns)

nullrates=holder[holder['Rating color'] == 'White'].groupby('Country').size().reset_index().rename(columns={0:'ratings'})

nullcountry = nullrates.Country.value_counts().index
nullcolor = nullrates.value_counts().values
plt.pie(nullrates['ratings'][2:],labels=nullrates['Country'][2:],autopct='%1.2f%%')
plt.show()
print(nullrates)
#print(nullcolor)

##find out which courrency used in every country
print(holder.columns)
notes = holder[['Country','Currency']].groupby(['Country','Currency']).size().reset_index().rename(columns={0:'used'})
plt.pie(notes['used'][10:12],labels=notes['Currency'][10:12],autopct = '%1.2f%%')
plt.show()
print(notes)

onlined = holder[holder['Has Online delivery']=='Yes'].groupby('Country').size().reset_index()
print(onlined)

cities_val= holder.City.value_counts().values
city_labels = holder.City.value_counts().index
plt.pie(cities_val[:5],labels=city_labels[:5],autopct = '%1.2f%%')
plt.show()

cuisines = holder.groupby('Cuisines').size().reset_index().rename(columns={0:'top_10'})

cuisines_sorted = cuisines.sort_values(by = 'top_10',ascending = False)

cuisines_top_10 = cuisines_sorted.head(10)
sns.barplot(x='top_10',y = 'Cuisines',data = cuisines_top_10,palette=['green','red'])
plt.show()
print(cuisines_top_10)

print(df.head())