import pickle , joblib

model = joblib.load("spam-classifier.pkl")

with open("spam-classifier-vectorizer.pkl" , "rb") as f:
    vectorizer = pickle.load(f)

### start
msg = [
    "Nokia"
]

X_input = vectorizer.transform(msg)
pred = model.predict(X_input)

for msg , rel in zip(msg , pred):
    print(f"message: {msg} -> {'spam' if rel == 1 else 'ham'}")


