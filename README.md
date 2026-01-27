Fake News Detection API 📰

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

Small and dated dataset

Binary classification only (REAL / FAKE)

English language only

No source credibility or fact verification

Not suitable for real-world fact-checking

Future Improvements

Larger and more diverse datasets

Transformer-based models (BERT, RoBERTa)

Explainability (SHAP / LIME)

Multi-class labels (satire, misleading, opinion)

Source credibility analysis