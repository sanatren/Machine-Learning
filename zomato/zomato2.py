import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('zomato.csv', encoding='latin-1')
df_country = pd.read_excel('Country-Code.xlsx')
holder = pd.merge(df, df_country, on='Country Code', how='left')

# Country-wise analysis
country_values = holder.Country.value_counts()
top_countries = country_values.index[:15]
top_country_values = country_values.values[:15]

# Rating-wise analysis
rating = holder.groupby(['Aggregate rating', 'Country Code', 'Rating color', 'Rating text']).size().reset_index(name='Rating_count')

# Null ratings analysis
nullrates = holder[holder['Rating color'] == 'White'].groupby('Country').size().reset_index().rename(columns={0:'ratings'})

# Currency analysis
notes = holder[['Country', 'Currency']].groupby(['Country', 'Currency']).size().reset_index(name='used')

# Online delivery analysis
onlined = holder[holder['Has Online delivery'] == 'Yes'].groupby('Country').size().reset_index()

# Cities analysis
cities_val = holder.City.value_counts()[:5]
city_labels = cities_val.index
city_values = cities_val.values

# Cuisines analysis
cuisines = holder.groupby('Cuisines').size().reset_index(name='top_10')
cuisines_sorted = cuisines.sort_values(by='top_10', ascending=False)
cuisines_top_10 = cuisines_sorted.head(10)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Pie charts
axes[0, 0].pie(top_country_values[1:5], labels=top_countries[1:5], autopct='%1.2f%%')
axes[0, 0].set_title('Top 5 Countries Using Zomato After india')
axes[0, 1].pie(nullrates['ratings'][2:], labels=nullrates['Country'][2:], autopct='%1.2f%%')
axes[0, 1].set_title('Distribution of Null Ratings by Country')
axes[1, 0].pie(notes['used'][10:12], labels=notes['Currency'][10:12], autopct='%1.2f%%')
axes[1, 0].set_title('Currency Usage by Country')
axes[1, 1].pie(cities_val, labels=city_labels, autopct='%1.2f%%')
axes[1, 1].set_title('Top 5 Cities')

plt.tight_layout()
plt.show()
plt.figure(figsize=(18, 12))

# Bar plot 1: Rating Distribution
axes[0,0]=sns.barplot(x='Aggregate rating', y='Rating_count', hue='Rating color', data=rating, palette=['silver', 'red', 'orange', 'yellow', 'green', 'green'])
axes[0,0]=plt.title('Rating Distribution')



# Bar plot 2: Top 10 Cuisines
plt.figure(figsize=(12, 6))
axes[0,1]=sns.barplot(x='top_10', y='Cuisines', data=cuisines_top_10, palette=['green', 'red'])
axes[0,1]=plt.title('Top 10 Cuisines')
plt.tight_layout()
plt.show()