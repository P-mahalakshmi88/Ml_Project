import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import pickle

# Download stopwords
nltk.download('stopwords')

# Load dataset
news_dataset = pd.read_csv('train.csv')

# Replace null values with empty strings
news_dataset = news_dataset.fillna('')

# Merge 'author' and 'title' columns
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Apply stemming
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separating features and label
X = news_dataset['content']
Y = news_dataset['label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Save the vocabulary for unseen token detection
trained_vocab = set(vectorizer.get_feature_names_out())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy scores
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model, vectorizer, and vocabulary using pickle
with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('trained_vocab.pkl', 'wb') as vocab_file:
    pickle.dump(trained_vocab, vocab_file)

print("Model, vectorizer, and vocabulary saved successfully.")
