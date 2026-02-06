import pandas as pd
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
#1.Loading the dataset
df=pd.read_csv('customer_churn_dataset.csv')
#2.Remove Duplicates and fill missing values with mean
df=df.drop_duplicates()
df=df.fillna(df.mean(numeric_only=True))
#3.Encoding categorical variables
df=pd.get_dummies(df,drop_first=True)

#4.Splitting the dataset into targeted variable
x=df.drop('churn',axis=1)
y=df['churn']
x_d=df.drop('tenure',axis=1)
y_d=df['tenure']
k=StandardScaler()
k.fit(x_d,y_d)
print(y_d)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#5.Train the model
model=LogisticRegression()
model.fit(x_train,y_train)

#6.predicting the test set results
y_pred=model.predict(x_test)
#Evaluate the model
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
classification=classification_report(y_test,y_pred)
print("Classification_model:",classification)

confusion=confusion_matrix(y_test,y_pred)
print("Confusion_matrix:",confusion)

