import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report ,confusion_matrix
df=pd.read_csv("customer_churn_dataset.csv")
print(df.info())
print(df.describe())
for  col in df.columns:
    if df[col] == type('objcet'):
       df[col]=pd.to_numeric(df,errors='coerce')
       df[col]=df[col].fillna(df[col].mode()[0])
    else:
        df[col]=df[col].fillna(df[col].median())
df=pd.get_dummies(df,drop_first=True)
target=df.columns[-1]
x=df.drop(target,axis=1)
y=df[target]
model=RandomForestClassifier()
model.fit()
x_train,y_train,x_test,y_test=train_test_split(df,test_size=0.2,random_state=40)
y_pd=model.predict(x_test)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
print(classification_report(y_pd,y_test))
print(confusion_matrix(y_pd,y_test))

