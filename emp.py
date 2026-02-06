import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("employee_salary_prediction.csv")
print(df)
x=df[['experience_years','education_level','job_role']]
y=df['salary']

df=pd.get_dummies(df,columns=['education_level','job_role'],drop_first=True)
print(df)
x=df.drop('salary',axis=1)
y=df['salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accruracy=accuracy_score(y_test,y_pred)
print("accruracy",y_pred)
confess_matrix=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(confess_matrix)

