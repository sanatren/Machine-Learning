#handling the tags section
#from sklearn.preprocessing import MultiLabelBinarizer
#import ast

#mlb = MultiLabelBinarizer()

#Converting string representation of lists to actual lists
#df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
#tags_encode = mlb.fit_transform(df['tags'])

# Convert the encoded tags into a DataFrame
#tags_encoded_df = pd.DataFrame(tags_encode,columns= mlb.classes_)
#tags_encoded_df = tags_encoded_df.astype(int)
# Concatenate the encoded tags DataFrame with the original DataFrame
#df = pd.concat([df, tags_encoded_df], axis=1)

#df.drop(columns=['tags'], inplace=True)