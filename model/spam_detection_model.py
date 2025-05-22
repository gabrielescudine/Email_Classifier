# This model will train a spam detection model using the Multinomial Naive Bayes algorithm.
# The division of the dataset will be 80% for training and 20% for testing.
# After that, the model will be saved in the model folder as spam_classifier_model.joblib
# and we will validate the model using synthetic data (soon implementation).

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset, features and target
dataset = pd.read_csv('../data/dataset.csv')
X = pd.read_csv('../data/X.csv')
y = pd.read_csv('../data/y.csv')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Training the model
model = MultinomialNB()
model.fit(X_train, y_train.values.ravel())

# Predicting the test set results
y_pred = model.predict(X_test)

# Printing the classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassifier Report:\n", classification_report(y_test, y_pred))

# Saving the model
model = joblib.dump(model, './spam_classifier_model.joblib')

print("Model saved as spam_classifier_model.joblib on the model folder.")