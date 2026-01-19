

# Twitter Sentiment Analysis Web App

## Overview

This project is a **web-based Twitter Sentiment Analysis application** built using **Streamlit** and a **deep learning CNN model**.
The application allows users to input a tweet and instantly predicts whether the sentiment is **positive or negative**.

The model is trained on preprocessed text data and deployed as an interactive web application, demonstrating the complete workflow from **model loading to real-time inference**.

---

## Application Features

* Interactive web interface using Streamlit
* Real-time sentiment prediction
* Deep learning–based text classification
* Clean and minimal user experience
* Model inference without retraining

---

## Tech Stack

* Python
* Streamlit
* TensorFlow / Keras
* Pickle
* NumPy

---

## Model Details

* **Architecture**: Convolutional Neural Network (CNN)
* **Input**: Tokenized and padded tweet text
* **Tokenizer**: Pre-trained tokenizer loaded using Pickle
* **Sequence Length**: 100 tokens
* **Output**:

  * Probability score
  * Threshold-based classification:

    * Positive sentiment if probability > 0.5
    * Negative sentiment otherwise

---

## Project Structure

```
twitter-sentiment-analyzer/
│
├── app.py                 # Streamlit application
├── cnn_sentiment.h5       # Trained CNN model
├── tokenizer.pkl          # Tokenizer for text preprocessing
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## How the Application Works

1. User enters a tweet in the text area
2. Text is converted to lowercase
3. Tokenizer transforms text into sequences
4. Sequences are padded to fixed length
5. CNN model predicts sentiment probability
6. Result is displayed on the UI

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## Use Cases

* Social media sentiment monitoring
* Brand perception analysis
* Public opinion tracking
* NLP model deployment demonstration

---

## Future Improvements

* Multi-class sentiment classification
* Support for emojis and hashtags
* Text cleaning pipeline (stopwords, lemmatization)
* Model explainability
* Deployment on cloud platforms

---

## Author

**Jai Kishan Kokkiligadda**
Data Science and Machine Learning

* Suggest **cloud deployment (AWS / GCP / Streamlit Cloud)**

