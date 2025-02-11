import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("ust_data_.csv")
data["email"] = data["head"] + data["name"] + data["address"] + data["sensitivity"] + data["importance"] + data["text"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["email"])

X_labeled = X[:215]
labeled_data = data[:215]
y_labeled = labeled_data["label"]

model = MultinomialNB()
model.fit(X_labeled , y_labeled)

X_unlabeled = X[215 : ]
y_pred = model.predict(X_unlabeled)

mylabel = data["label"].to_numpy()
new_arr = np.append(mylabel[:215] , y_pred)

email_new = pd.DataFrame()
email_new["label"] = pd.Series(new_arr)
email_new["email"] = data["email"]

email_new.to_csv("clean_ust_data.csv" , index=False)





