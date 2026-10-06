import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("spam.csv")

#Duplicates
df = df.drop_duplicates()

messages = df["Message"].tolist()
category = df["Category"].tolist()

pattern = r"[a-z0-9]+(?:'[a-z]+)*"

def tokenize(text):
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    tokens = re.findall(pattern, text)
    return tokens

tokenized = []
for message in messages:
    tokens = tokenize(message)
    tokenized.append(tokens)

unique_tokens = set()
for line in tokenized:
    for token in line:
        unique_tokens.add(token)

unique_tokens = sorted(unique_tokens)
vocab = {}
for index, token in enumerate(unique_tokens):
    vocab[token] = index

matrix = []
for message_tokens in tokenized:
    row = [0] * len(vocab)
    for token in message_tokens:
        col = vocab[token]
        row[col] = 1
    matrix.append(row)

x = np.array(matrix, dtype=np.int8)

y_list = []
for c in category:
    if c == "spam":
        y_list.append(1)
    else:
        y_list.append(0)

y = np.array(y_list)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Results on the test set")
print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-score:  {f1:.2%}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nConfusion matrix")
print(f"                 pred. ham   pred. spam")
print(f"actual ham       {tn:>9}   {fp:>10}")
print(f"actual spam      {fn:>9}   {tp:>10}")