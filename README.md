# 🎬 Movie Review Sentiment Analysis

This is a **Flask-based web application** for sentiment analysis of movie reviews, leveraging a deep learning model trained on the IMDB dataset. The model combines **Convolutional Neural Networks (CNN)** and **Bidirectional LSTMs** with **GloVe embeddings** to predict whether a review is **positive** or **negative**.

---

## 🚀 Features

- **Sentiment Prediction**: Classifies movie reviews as Positive 😊 or Negative 😔 with confidence scores.
- **Input Preprocessing**: Normalizes text by removing punctuation, fixing spaces, and capitalizing the first word.
- **Spell Checking**: Uses `pyspellchecker` to detect and suggest corrections for misspelled words.
- **User-Friendly Interface**: Modern, responsive UI with real-time word count feedback and clickable spelling suggestions.
- **Minimum Word Requirement**: Ensures reviews have at least 15 words for reliable predictions.

---

## 🧠 Model Details

**Architecture:**
- Embedding layer with pre-trained **GloVe (100D)** embeddings.
- `Conv1D` with 64 filters, followed by max pooling.
- Two Bidirectional LSTM layers (128 and 64 units) with dropout (0.3).
- Dense layer with sigmoid activation for binary classification.

**Training:**
- Trained on the **IMDB dataset** (5000-word vocabulary, 500-word max length).

**Performance:**
- ✅ Accuracy: ~80.66%
- 🧪 Precision: 78.35%
- 🔁 Recall: 84.74%
- 📊 F1-score: 81.42%

📁 **Source**: Model training code is available in `setimemt_analysis_imdb.ipynb`.

---

## 🛠 Prerequisites

- Python 3.7
- Pre-trained model file: `sentiment_analysis_model_improved.h5`
- Flask development server (for local testing)

---

## 📦 Installation

### 1. Clone the Repository
```bash
  git clone https://github.com/Sachin22424/Sentiment_analysis_web_app
  cd Sentiment_analysis_web_app
```
### 2. Create a Virtual Environment
```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
  pip install -r requirements.txt
```
---

## 🧪 Usage
Run the Application
```bash
  python app.py
```

The app will start at: http://127.0.0.1:5000

Use the Web Interface:
Open your browser and navigate to http://127.0.0.1:5000

Enter a movie review (minimum 15 words) in the text area.

Click "Analyze Sentiment" to see the prediction and confidence score.

