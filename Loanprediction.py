import pandas as pd

df = pd.read_csv("https://github.com/ZagaUS/zaga-nec-hackathon-2/blob/bb0450dda2c89c3709a7701de978542de8b1276f/loanprediction.csv")

print("Dataset loaded successfully", df)

# print("\nMissing values:")
# print(df.isnull().sum())

# df.fillna(df.mean(numeric_only=True), inplace=True)


# df = pd.get_dummies(df)

# X = df.drop("LoanPrediction", axis=1)
# y = df["LoanPrediction"]


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.2, random_state=42)


# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)


# from sklearn.metrics import accuracy_score
# pred = model.predict(X_test)
# print("\nModel Accuracy:", accuracy_score(y_test, pred))
