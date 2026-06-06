import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "spam_dataset.csv")

df = pd.read_csv(file_path)

encoder = LabelEncoder()

y = encoder.fit_transform(df["label"])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])

X_train, X_test, y_train, y_test = train_test_split (X,y,test_size=0.2,random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

print(X_test)

predictions = model.predict(X_test)

print(predictions)