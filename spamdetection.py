import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Load the dataset
df=pd.read_csv('spam_data.csv')
print(df.head())
#Handle missing or incomplete records
df['email_text']=df['email_text'].dropna()
#Train a suitable classification model
X=df['email_text']
Y=df['num_words']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
vectorizer=TfidfVectorizer()
x_train_vector=vectorizer.fit_transform(x_train)
y_train_vector=vectorizer.fit_transform(y_train)
x_testvector=vectorizer.transform(x_test)
y_testvector=vectorizer.transform(y_test)
model=LogisticRegression()
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print(f'Model Accuracy:{accuracy*100:.2f}%')







