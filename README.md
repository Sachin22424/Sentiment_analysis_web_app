Sentiment Analysis Web Application
Overview
This project is a Flask-based web application for sentiment analysis, predicting whether a given text input (e.g., movie review) is positive or negative. The application uses a pre-trained CNN-BiLSTM model with GloVe embeddings, trained on the IMDB dataset, to classify sentiment. The frontend is built with Tailwind CSS and custom styles, featuring real-time input validation, a confidence bar, and emoji-based output (😊 for positive, 😔 for negative). The backend robustly preprocesses inputs, handling capitalization, extra spaces, punctuation, and non-ASCII characters.
The application is designed to be user-friendly, accessible, and robust, with features like:

Real-time word count and input validation.
Error messages with suggestions for correction.
Display of cleaned input to show preprocessing effects.
Animated UI with a modern gradient design.

Features

Sentiment Prediction: Classifies text as Positive (😊) or Negative (😔) with a confidence score.
Robust Preprocessing:
Normalizes spaces and converts to lowercase.
Removes punctuation and non-ASCII characters (e.g., emojis).
Maps words to IMDB word indices with proper offset handling.
Uses NLTK for accurate tokenization.


Frontend Enhancements:
Real-time input validation (green/red border for valid/invalid input).
Confidence bar for visual feedback.
Dismissible error alerts and a "Try Another" button.
Accessible with ARIA labels and clear placeholders.


Error Handling:
Requires ≥3 words for prediction.
Warns for short inputs (<5 words) that may reduce accuracy.
Suggests fixes for invalid inputs (e.g., special characters).



Project Structure
sentiment-analysis-webapp/
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # HTML template for frontend
├── static/
│   └── styles.css                  # Custom CSS
├── sentiment_analysis_model_improved.h5  # Pre-trained model
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies

Prerequisites

Python: Version 3.7.9 (specified for compatibility with TensorFlow 2.10.0).
Virtual Environment: Recommended for dependency isolation.
Model File: sentiment_analysis_model_improved.h5 (pre-trained CNN-BiLSTM model).
Internet: For Tailwind CSS CDN and NLTK data download.

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd sentiment-analysis-webapp


Create a Virtual Environment:
python3.7 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


Install Dependencies:
pip install -r requirements.txt


Download NLTK Data:
python -m nltk.downloader punkt


Place the Model File:

Ensure sentiment_analysis_model_improved.h5 is in the project root.
If unavailable, train the model using the original notebook and save it as sentiment_analysis_model_improved.h5.


Run the Application:
python app.py


Open http://127.0.0.1:5000 in a web browser (Chrome/Firefox recommended).



Usage

Enter a Review:

Type a review in the textarea (e.g., “This movie was fantastic!”).
Minimum 3 words required; ≥5 words recommended for accurate predictions.
Avoid special characters (#, @, emojis) for best results.


Real-Time Validation:

The textarea border turns green (≥3 words) or red (<3 words).
Word count and special character warnings display below the textarea.


Analyze Sentiment:

Click “Analyze Sentiment” to predict.
Results show sentiment (Positive 😊 or Negative 😔), confidence (%), and a confidence bar.
Cleaned input is displayed to show preprocessing effects.
Short reviews (<5 words) trigger a warning.


Try Another:

Click “Try Another” to reset the form and enter a new review.



Example Inputs and Outputs

Input: “This movie was fantastic!”  
Output: Positive 😊, Confidence: ~85–95%, Cleaned: “this movie was fantastic”


Input: “Terrible plot, so boring.”  
Output: Negative 😔, Confidence: ~80–90%, Cleaned: “terrible plot so boring”


Input: “Great film!”  
Output: Positive 😊, Confidence: ~70–80%, Cleaned: “great film”, Warning: Short review


Input: “@movie #bad”  
Output: Error: Input must contain at least 3 words



Training the Model
The model (sentiment_analysis_model_improved.h5) is a CNN-BiLSTM with GloVe embeddings, trained on the IMDB dataset. To retrain:

Use the original Jupyter notebook (not included here).
Key steps:
Load IMDB dataset (top_words=5000, max_review_length=500).
Use GloVe embeddings (glove.6B.100d.txt).
Train with Adam optimizer, focal loss, and early stopping.
Save as sentiment_analysis_model_improved.h5.


Place the saved model in the project root.

Troubleshooting

Model Loading Error:
Ensure sentiment_analysis_model_improved.h5 is in the project root.
Verify TensorFlow 2.10.0 (pip show tensorflow).


Inaccurate Predictions:
Check raw_output in the UI (near 0.5 indicates ambiguity).
Use movie-related, English reviews similar to IMDB data.
Uncomment sequence logging in app.py to debug preprocessing.


NLTK Errors:
Run python -m nltk.downloader punkt if tokenization fails.


UI Issues:
Verify Tailwind CSS CDN connectivity.
Check JavaScript console (F12 → Console) for errors.


Flask Errors:
Ensure port 5000 is free (lsof -i :5000 or netstat -aon).
Reinstall Flask (pip install flask).



Future Improvements

Model:
Fine-tune on diverse datasets (e.g., Sentiment140) for better generalization.
Use DistilBERT for improved short-text performance.


Backend:
Add a REST API endpoint for programmatic access.
Store predictions in a SQLite database.


Frontend:
Add client-side input cleaning before submission.
Support multi-class sentiment (positive, neutral, negative).



Technologies Used

Backend: Flask, TensorFlow 2.10.0, NumPy, NLTK
Frontend: Tailwind CSS, HTML, JavaScript
Model: CNN-BiLSTM with GloVe embeddings
Environment: Python 3.7.9

License
MIT License. See LICENSE file (if included) for details.
Contact
For issues or suggestions, open a GitHub issue or contact [your-email@example.com].
