import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Enhanced text preprocessing for better model accuracy"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove very short words (less than 2 characters) except common ones
    words = text.split()
    common_short_words = ['a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we']
    words = [w for w in words if len(w) > 2 or w in common_short_words]
    text = ' '.join(words)
    
    return text

def train_improved_model():
    """Train an improved model with better preprocessing"""
    print("=" * 70)
    print("🚀 IMPROVING MODEL ACCURACY")
    print("=" * 70)
    
    try:
        # Load the dataset
        print("\n📂 Loading dataset...")
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
        
        # Add labels
        true_df['label'] = 1  # Real news
        fake_df['label'] = 0   # Fake news
        
        # Combine datasets
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"✅ Dataset loaded: {len(df)} articles")
        print(f"   - Real news: {len(true_df)}")
        print(f"   - Fake news: {len(fake_df)}")
        
        # Preprocess text
        print("\n🔧 Preprocessing text data...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 10]
        print(f"✅ After preprocessing: {len(df)} articles")
        
        # Prepare data
        X = df['processed_text']
        y = df['label']
        
        # Split data
        print("\n📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create improved vectorizer with better parameters
        print("\n🔧 Creating enhanced TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=20000,  # Increased features
            ngram_range=(1, 3),  # Include trigrams
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # Use logarithmic scaling
            norm='l2'  # L2 normalization
        )
        
        # Fit vectorizer
        print("📐 Fitting vectorizer...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train improved model
        print("\n🤖 Training improved Logistic Regression model...")
        model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train_tfidf, y_train)
        
        # Test the model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Detailed metrics
        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANCE METRICS")
        print("=" * 70)
        print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 CONFUSION MATRIX:")
        print("                 Predicted")
        print("               Fake  Real")
        print(f"Actual Fake    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Real    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Test with sample texts
        print("\n" + "=" * 70)
        print("🧪 TESTING WITH SAMPLE TEXTS")
        print("=" * 70)
        sample_texts = [
            "Scientists discover new planet with potential for life in nearby star system",
            "BREAKING: ALIENS LANDED IN MY BACKYARD!!! YOU WON'T BELIEVE WHAT HAPPENED NEXT!!!",
            "Government announces new economic policies to boost growth",
            "SHOCKING: Local man wins lottery with numbers from a dream!",
            "Research study shows benefits of exercise on mental health",
            "URGENT: Click here now to claim your prize! Don't miss out!"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            processed = preprocess_text(text)
            text_tfidf = vectorizer.transform([processed])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities) * 100
            
            result = "REAL" if prediction == 1 else "FAKE"
            print(f"\n   {i}. \"{text[:50]}...\"")
            print(f"      → {result} (Confidence: {confidence:.1f}%)")
        
        # Save the improved model and vectorizer
        print("\n" + "=" * 70)
        print("💾 SAVING IMPROVED MODEL")
        print("=" * 70)
        print("\nSaving model and vectorizer...")
        joblib.dump(model, "lr_model.jb")
        joblib.dump(vectorizer, "vectorizer.jb")
        
        print("✅ Improved model saved successfully!")
        print(f"\n🎉 Model improvement complete!")
        print(f"   Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Improvements:")
        print(f"   - Enhanced text preprocessing (URL removal, normalization)")
        print(f"   - Better TF-IDF parameters (trigrams, more features)")
        print(f"   - Class balancing for better fake news detection")
        print(f"   - Improved model hyperparameters")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Dataset files not found: {e}")
        print("   Please ensure 'True.csv' and 'Fake.csv' are in the current directory")
        return False
    except Exception as e:
        print(f"\n❌ Error improving model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_improved_model()

