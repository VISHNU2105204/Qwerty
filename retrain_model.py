import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def retrain_model():
    """Retrain the model with current scikit-learn version"""
    print("Retraining Model with Current Scikit-learn Version...")
    print("=" * 60)
    
    try:
        # Load the dataset
        print("Loading dataset...")
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
        
        # Add labels
        true_df['label'] = 1  # Real news
        fake_df['label'] = 0   # Fake news
        
        # Combine datasets
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        print(f"Dataset size: {len(df)} articles")
        print(f"   - Real news: {len(true_df)}")
        print(f"   - Fake news: {len(fake_df)}")
        
        # Prepare data
        X = df['text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create new vectorizer
        print("\nCreating new vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Fit vectorizer
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train new model
        print("Training new model...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # Test the new model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nNew Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Test with sample texts
        print("\nTesting with sample texts:")
        sample_texts = [
            "Scientists discover new planet with potential for life",
            "Breaking news: Local man wins lottery", 
            "ALIENS LANDED IN MY BACKYARD!!!",
            "Government announces new economic policies",
            "You won't believe what happened next!"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            text_tfidf = vectorizer.transform([text])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            result = "REAL" if prediction == 1 else "FAKE"
            print(f"   {i}. \"{text[:40]}...\" -> {result} (Confidence: {confidence:.2f})")
        
        # Save the new model and vectorizer
        print("\nSaving new model...")
        joblib.dump(model, "lr_model.jb")
        joblib.dump(vectorizer, "vectorizer.jb")
        
        print("New model saved successfully!")
        print("Version compatibility issue fixed!")
        
        return True
        
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False

if __name__ == "__main__":
    retrain_model()
