from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import re
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker  

# Download NLTK data
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Initialize spell checker
spell = SpellChecker()

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
min_words = 15

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(text)
    misspelled = spell.unknown(words)
    corrections = {word: spell.correction(word) for word in misspelled if spell.correction(word)}
    if words:
        words[0] = words[0].capitalize()
    sequence = []
    for word in words:
        idx = word_index.get(word.lower(), 0)
        if idx > 0 and idx < top_words:
            adjusted_idx = idx + 3
            if adjusted_idx < top_words:
                sequence.append(adjusted_idx)
        else:
            sequence.append(2)
    sequence = pad_sequences([sequence], maxlen=max_review_length, padding='pre', truncating='pre')
    return sequence, corrections, words

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    emoji = None
    error = None
    corrections = None
    word_count = 0
    review_text = ''

    if request.method == 'POST':
        review = request.form.get('review')
        review_text = review  # Preserve the input text
        if not review or len(review.strip()) < 10:
            error = "Please enter a review with at least 10 characters."
        else:
            try:
                sequence, corrections, words = preprocess_text(review)
                word_count = len(words)
                if word_count < min_words:
                    error = f"Review is too short ({word_count} words). Please use at least {min_words} words for accurate results."
                else:
                    prediction = model.predict(sequence, verbose=0)[0][0]
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
                    emoji = '😊' if sentiment == 'Positive' else '😔'
                    confidence = min(confidence, 0.9999)
                    if corrections:
                        error = "Possible spelling errors detected. See suggestions below."
            except Exception as e:
                error = f"Error processing review: {str(e)}"

    return render_template('index.html', sentiment=sentiment, confidence=confidence, emoji=emoji, error=error, corrections=corrections, word_count=word_count, min_words=min_words, review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True)