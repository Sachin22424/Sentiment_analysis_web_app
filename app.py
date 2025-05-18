from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Load the saved model
try:
    model = tf.keras.models.load_model('sentiment_analysis_model_improved.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load IMDB word index for preprocessing
word_index = imdb.get_word_index()
max_review_length = 500
top_words = 5000

def preprocess_text(text):
    # Clean text: normalize spaces, remove non-ASCII, punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII (e.g., emojis)
    text = re.sub(r'[^\w\s]', '', text.lower().strip())  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    # Tokenize using NLTK
    words = word_tokenize(text)
    if len(words) < 3:
        return None, text, "Input must contain at least 3 words."
    # Map words to IMDB indices, adjusting for offset
    sequence = []
    for word in words:
        idx = word_index.get(word, 0)
        if idx > 0 and idx < top_words:
            adjusted_idx = idx + 3
            if adjusted_idx < top_words:
                sequence.append(adjusted_idx)
        else:
            sequence.append(2)  # Unknown words
    # Pad sequence
    sequence = pad_sequences([sequence], maxlen=max_review_length, padding='pre', truncating='pre')
    # print("Preprocessed sequence:", sequence)  # Debug
    return sequence, text, None

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    emoji = None
    error = None
    raw_output = None
    cleaned_input = None

    if request.method == 'POST':
        review = request.form.get('review')
        if not review or len(review.strip()) == 0:
            error = "Please enter a valid review."
        else:
            try:
                sequence, cleaned_input, preprocess_error = preprocess_text(review)
                if preprocess_error:
                    error = preprocess_error
                else:
                    prediction = model.predict(sequence, verbose=0)[0][0]
                    raw_output = float(prediction)
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
                    emoji = '😊' if sentiment == 'Positive' else '😔'
                    confidence = min(confidence, 0.9999)
                    if len(word_tokenize(cleaned_input)) < 5:
                        error = "Short review (<5 words). Results may be less accurate."
            except Exception as e:
                error = f"Error processing review: {str(e)}. Try using clear English text without special characters."

    return render_template('index.html', sentiment=sentiment, confidence=confidence, emoji=emoji, error=error, raw_output=raw_output, cleaned_input=cleaned_input)

if __name__ == '__main__':
    app.run(debug=True)