import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("emails.csv")
df.columns = ["text" , "label"]
###
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_t , X_test , y_t , y_test = train_test_split(X , y , test_size=0.1 ,random_state=42)

model = MultinomialNB(alpha=1.0e-10).fit(X_t , y_t)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print(accuracy)

new = [
    """
You're signed up to receive a weekly report of some notifications from your Canvas account. Below is the report for the week ending Feb 9:

Assignment Created - Assignment 1, MATH1014 (L01-L12) - Calculus II

due: Feb 20 at 11:59pm
"""
]

pred = model.predict(vectorizer.transform(new))

if pred == 0:
    print("ham")
else:
    print("spam")




