"""
Prediction Service for Fake News Detection
Handles model loading and prediction logic
"""

import pickle
import os
from utils.text_preprocessing import stemming



class PredictionService:
    
    
    def __init__(self):
        """Initialize the service by loading model and vectorizer"""
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
      
        try:
            # Define paths to model files
            model_path = os.path.join("model", "fakenews_model.pkl")
            vectorizer_path = os.path.join("model", "vectorizer.pkl")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            
            # Load model
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✓ Model loaded successfully")
            
            # Load vectorizer
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print("✓ Vectorizer loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str) -> dict:
        """
        Predict if a news article is REAL or FAKE.
        
        Args:
            text (str): The news article text/title to classify
            
        Returns:
            dict: Contains prediction, label, and confidence score
            
        Example:
            >>> service = PredictionService()
            >>> result = service.predict("Breaking news about politics")
            >>> print(result)
            {'prediction': 'FAKE', 'label': 0, 'confidence': 0.85}
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text input cannot be empty")
        
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Step 1: Clean the text using the same preprocessing as training
        cleaned_text = stemming(text)
        
        # Step 2: Convert text to numerical features using vectorizer
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        # Step 3: Make prediction
        prediction = self.model.predict(vectorized_text)[0]
        
        # Step 4: Get prediction probability (confidence)
        prediction_proba = self.model.predict_proba(vectorized_text)[0]
        
        # Get confidence for the predicted class
        if prediction == "REAL":
            confidence = prediction_proba[1]  # Probability of REAL class
            label = 1
        else:
            confidence = prediction_proba[0]  # Probability of FAKE class
            label = 0
        
        # Return structured result
        return {
            "prediction": prediction,
            "label": label,
            "confidence": float(confidence),
            "cleaned_text": cleaned_text  # Include for debugging
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict multiple news articles at once.
        
        Args:
            texts (list): List of news article texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "text": text
                })
        return results


# Create a singleton instance
prediction_service = PredictionService()