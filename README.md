Movie Review Sentiment Analysis
This is a Flask-based web application for sentiment analysis of movie reviews, leveraging a deep learning model trained on the IMDB dataset. The model uses a combination of Convolutional Neural Networks (CNN) and Bidirectional LSTMs with GloVe embeddings to predict whether a review is positive or negative.
Features

Sentiment Prediction: Classifies movie reviews as Positive 😊 or Negative 😔 with confidence scores.
Input Preprocessing: Normalizes text by removing punctuation, fixing spaces, and capitalizing the first word.
Spell Checking: Uses pyspellchecker to detect and suggest corrections for misspelled words.
User-Friendly Interface: Modern, responsive UI with real-time word count feedback and clickable spelling suggestions.
Minimum Word Requirement: Ensures reviews have at least 15 words for reliable predictions.

Model Details

Architecture: 
Embedding layer with pre-trained GloVe (100D) embeddings.
Conv1D with 64 filters, followed by MaxPooling.
Two Bidirectional LSTM layers (128 and 64 units) with dropout (0.3).
Dense layer with sigmoid activation for binary classification.


Training: Trained on the IMDB dataset (5000-word vocabulary, 500-word max length).
Performance: Achieves ~80.66% accuracy, 78.35% precision, 84.74% recall, and 81.42% F1-score on the test set.
Source: Model training code is available in setimemt_analysis_imdb.ipynb.

Prerequisites

Python 3.8+
A pre-trained model file (sentiment_analysis_model_improved.h5) from the training notebook.
Flask development server for local testing.

Installation

Clone the Repository:
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the Model:

Ensure sentiment_analysis_model_improved.h5 is in the project root. You can generate it by running the provided Jupyter notebook (setimemt_analysis_imdb.ipynb) or download it from [your model source].


Directory Structure:
movie-review-sentiment-analysis/
├── app.py
├── sentiment_analysis_model_improved.h5
├── static/
│   └── styles.css
├── templates/
│   └── index.html
├── setimemt_analysis_imdb.ipynb
├── requirements.txt
└── README.md



Usage

Run the Application:
python app.py

The app will start at http://127.0.0.1:5000.

Access the Web Interface:

Open your browser and navigate to http://127.0.0.1:5000.
Enter a movie review (minimum 15 words) in the textarea.
Click "Analyze Sentiment" to see the predicted sentiment and confidence score.
If spelling errors are detected, suggestions will appear below the form, clickable to auto-correct.


Example Input:
This movie was absolutely fantastic with great acting and a compelling storyline that kept me engaged throughout.

Output: Positive 😊, Confidence: 92.34%


Dependencies
See requirements.txt for a complete list. Key dependencies include:

Flask: Web framework for the application.
TensorFlow: For loading and running the trained model.
NLTK: For text tokenization.
pyspellchecker: For spell-checking and suggestions.