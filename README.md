## Fake News Detection API 

## Live App

**Interactive Demo (Streamlit UI)**  
➜ https://fake-news-detection-ui.onrender.com/

**API Endpoint (FastAPI backend)**  
➜ https://coffee-sales-ml.onrender.com  
- OpenAPI docs: https://fake-news-detection-fastapi-e0xk.onrender.com/docs
- Health check: https://fake-news-detection-fastapi-e0xk.onrender.com/

A machine learning–based system that classifies news articles as REAL or FAKE using NLP and Logistic Regression.
The backend is built with FastAPI, and the frontend uses Streamlit.

Overview

This project demonstrates a complete ML pipeline:

Text preprocessing and feature extraction

Model training and evaluation

REST API deployment

Interactive web interface

It is intended for educational and portfolio purposes, not production fact-checking.

Key Features

Backend (FastAPI)

RESTful API with /predict and /health endpoints

Input validation and structured responses

Confidence score returned with each prediction

Automatic API documentation (Swagger / ReDoc)

Frontend (Streamlit)

Simple and clean UI

Real-time prediction results

Confidence visualization

Backend health status check

Machine Learning

TF-IDF vectorization

Logistic Regression classifier

Text preprocessing: cleaning, stopword removal, stemming

~83% accuracy on test data 
Project Structure:
fakenews-detection-api/
├── app.py
├── train_model.py
├── requirements.txt
├── frontend/
│   └── streamlit_app.py
├── schemas/
├── services/
├── utils/
├── model/
│   ├── fakenews_model.pkl
│   └── vectorizer.pkl
└── README.md
Model Details

Algorithm: Logistic Regression

Vectorization: TF-IDF

Dataset size: ~6,300 articles

Accuracy: ~90% (train), ~83% (test)

Limitations
1. Small Training Dataset

Only ~6,335 articles used for training
May not generalize well to diverse news topics
Limited to vocabulary seen during training

2. Simple Model Architecture

Single algorithm (Logistic Regression)
No deep learning or transformer models
No contextual understanding (unlike BERT, GPT)

3. No Source Verification

Doesn't check the credibility of news sources
Doesn't verify facts against databases
No cross-referencing with trusted media

4. Language Support

English only
No multilingual support

5. Temporal Bias

Trained on older news articles
May not detect modern misinformation tactics
No adaptation to evolving fake news patterns

6. Binary Classification Only

Only outputs FAKE or REAL
No nuance (partially true, misleading, satire, opinion)
No detection of clickbait or sensationalism


Future Improvements
To make this production-ready, consider:

Larger Dataset

Use datasets with 100K+ articles
Include diverse topics (politics, health, tech, sports)
Regularly update with fresh data


Advanced Models

Implement transformer models (BERT, RoBERTa)
Ensemble methods (combine multiple models)
Deep learning architectures (LSTM, CNN)


Additional Features

Source credibility scoring
Claim verification against fact-check databases
Sentiment analysis
Readability metrics
Image/video analysis (deepfake detection)


Multi-class Classification

Detect satire, opinion, clickbait
Identify partially true claims
Flag misleading headlines


Explainability

Show which words influenced the prediction
LIME or SHAP values
Highlight suspicious phrases


Real-time Learning

Continuous model updates
Feedback loop from users
A/B testing different models
