import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib
import pickle


df = pd.read_csv("data.csv" , encoding="latin-1" , usecols=[0,1])
df.columns = ["label" , "msg"]
df["label"] = df["label"].map({"ham" : 0 , "spam" : 1})

###
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["msg"])
Y = df["label"]
###

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)

model = MultinomialNB(alpha=1.0)
model.fit(X_train , Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test , Y_pred)
print(f"Accuracy is {accuracy}")

###
joblib.dump(model , "spam-classifier.pkl")

with open("spam-classifier-vectorizer.pkl" , "wb") as f:
    pickle.dump(vectorizer , f)

new_messages = ["Wanna try my new shirt"]
X_new = vectorizer.transform(new_messages)
pred = model.predict(X_new)

for msg, pred in zip(new_messages , pred):
    print(f"{msg} -> {'spam' if pred == 1 else 'ham'}")






