import pandas as pd
df=pd.read_csv("sentiment_analyze.csv")
print(df.head())
from textblob import TextBlob
def get_sentiment(text):
    blob=TextBlob(text)
    if blob.sentiment.polarity>0:
        return "positive"
    elif blob.sentiment.polarity<0:
        return "negative"
    else:
        return "neutral"
    df["sentiment"]=df["text"].apply(get_sentiment)
print(df.head())
df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)
print(df.head())