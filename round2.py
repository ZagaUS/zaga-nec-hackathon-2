from asyncio import Handle
import pandas as pd
df=pd.read_csv('sales_forecasting_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())    
print(df.isnull().sum())
df['Sales'].fillna(df['Sales'].mean(), inplace=True)  

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X = df.drop('Sales', axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Missing values in each column:")
print(df.isnull().sum())
















