from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask App
app = Flask(__name__)

# Load the trained model, vectorizer, and vocabulary
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('trained_vocab.pkl', 'rb') as vocab_file:
    trained_vocab = pickle.load(vocab_file)

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Stemming function (must match `model.py`)
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Function to detect if new news is fake based on unseen tokens
def is_new_news_fake(news_text, trained_vocab):
    preprocessed_text = stemming(news_text)
    tokens = set(preprocessed_text.split())
    unseen_tokens = tokens - trained_vocab

    if len(unseen_tokens) / len(tokens) > 0.5:  # If more than 50% tokens are unseen
        return "Fake News"
    else:
        return "Real News"

# Define Routes
@app.route('/')
def home():
    return render_template('home.html')  # HTML form for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (text) from user
        input_data = request.get_json()
        input_news = input_data['content']

        # Preprocess input text
        preprocessed_text = stemming(input_news)
        input_features = vectorizer.transform([preprocessed_text])

        # Check for new news
        new_news_result = is_new_news_fake(input_news, trained_vocab)

        # Predict with model if not classified as new news
        if new_news_result == "Real News":
            prediction = model.predict(input_features)
            result = 'Fake News' if prediction[0] == 1 else 'Real News'
        else:
            result = new_news_result

        return jsonify({'status': 'success', 'response': result})
    except Exception as e:
        return jsonify({'status': 'error', 'response': str(e)})

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
