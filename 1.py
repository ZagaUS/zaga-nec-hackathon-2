import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
df=pd.read_csv("movie_ratings_data.csv")
print(df.info())
print(df.head())
print(df.shape)
df=df.drop_duplicates()
df.fillna(df.mean(numeric_only=True),inplace=True)
df=pd.get_dummies(df, drop_first=True).astype(int)
print(df)
target=df.columns[-1]
X=df.drop(target , axis=1)
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("Acc:",accuracy_score(y_test,pred))
