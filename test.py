import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('spam.csv')

# Preprocess the data 
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Save the model
import joblib
joblib.dump(clf, 'spam_detector.joblib')

# Use the model to detect spam in new emails
def detect_spam(email):
    vector = vectorizer.transform(email)
    prediction = clf.predict(vector)
    return prediction[0]

# Example usage
email = ['Get the work done, work done']
print(detect_spam(email))