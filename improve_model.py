import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    words = [w for w in words if len(w) > 2 or w in ['a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we']]
    text = ' '.join(words)
    
    return text

def extract_features(text):
    """Extract additional features that help detect fake news"""
    features = {}
    
    # Count exclamation marks (fake news often has more)
    features['exclamation_count'] = text.count('!')
    
    # Count question marks
    features['question_count'] = text.count('?')
    
    # Count all caps words (fake news often uses ALL CAPS)
    words = text.split()
    features['all_caps_count'] = sum(1 for w in words if w.isupper() and len(w) > 1)
    
    # Count numbers (fake news might have more specific numbers)
    features['number_count'] = len(re.findall(r'\d+', text))
    
    # Average word length
    if words:
        features['avg_word_length'] = np.mean([len(w) for w in words])
    else:
        features['avg_word_length'] = 0
    
    # Text length
    features['text_length'] = len(text)
    
    # Word count
    features['word_count'] = len(words)
    
    # Count sensational words
    sensational_words = ['shocking', 'amazing', 'unbelievable', 'incredible', 'you won\'t believe', 
                        'breaking', 'urgent', 'alert', 'warning', 'exclusive', 'revealed', 'exposed']
    features['sensational_count'] = sum(1 for word in sensational_words if word in text.lower())
    
    return features

def prepare_enhanced_features(df):
    """Prepare enhanced features for training"""
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df['processed_text'].str.len() > 10]
    
    print("Extracting additional features...")
    feature_list = []
    for text in df['processed_text']:
        features = extract_features(text)
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    
    return df, feature_df

def train_improved_model():
    """Train an improved model with better preprocessing and ensemble methods"""
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
        
        # Prepare enhanced features
        df, feature_df = prepare_enhanced_features(df)
        
        # Prepare data
        X_text = df['processed_text']
        y = df['label']
        
        # Split data
        print("\n📊 Splitting data...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split feature_df accordingly
        train_indices = X_train_text.index
        test_indices = X_test_text.index
        X_train_features = feature_df.loc[train_indices].reset_index(drop=True)
        X_test_features = feature_df.loc[test_indices].reset_index(drop=True)
        X_train_text = X_train_text.reset_index(drop=True)
        X_test_text = X_test_text.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
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
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)
        
        # Combine TF-IDF features with extracted features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf, X_train_features.values])
        X_test_combined = hstack([X_test_tfidf, X_test_features.values])
        
        # Train multiple models
        print("\n🤖 Training models...")
        
        # Model 1: Improved Logistic Regression
        print("   Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
        lr_model.fit(X_train_combined, y_train)
        lr_pred = lr_model.predict(X_test_combined)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"   ✅ Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
        
        # Model 2: Random Forest
        print("   Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train_combined, y_train)
        rf_pred = rf_model.predict(X_test_combined)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        
        # Choose the best model
        if lr_accuracy >= rf_accuracy:
            print(f"\n📌 Selected Logistic Regression (Best Accuracy: {lr_accuracy:.4f})")
            best_model = lr_model
            best_pred = lr_pred
        else:
            print(f"\n📌 Selected Random Forest (Best Accuracy: {rf_accuracy:.4f})")
            best_model = rf_model
            best_pred = rf_pred
        
        # Detailed metrics
        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANCE METRICS")
        print("=" * 70)
        accuracy = accuracy_score(y_test, best_pred)
        precision = precision_score(y_test, best_pred)
        recall = recall_score(y_test, best_pred)
        f1 = f1_score(y_test, best_pred)
        
        print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, best_pred, target_names=['Fake', 'Real']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_pred)
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
            features = extract_features(processed)
            feature_array = np.array([list(features.values())])
            combined = hstack([text_tfidf, feature_array])
            
            prediction = best_model.predict(combined)[0]
            probabilities = best_model.predict_proba(combined)[0]
            confidence = max(probabilities) * 100
            
            result = "REAL" if prediction == 1 else "FAKE"
            print(f"\n   {i}. \"{text[:50]}...\"")
            print(f"      → {result} (Confidence: {confidence:.1f}%)")
        
        # Save the improved model and vectorizer
        print("\n" + "=" * 70)
        print("💾 SAVING IMPROVED MODEL")
        print("=" * 70)
        print("\nSaving model and vectorizer...")
        joblib.dump(best_model, "lr_model.jb")
        joblib.dump(vectorizer, "vectorizer.jb")
        
        # Save preprocessing function info (we'll use it in server)
        print("✅ Improved model saved successfully!")
        print(f"\n🎉 Model improvement complete!")
        print(f"   Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Improvement: Better preprocessing, feature engineering, and model selection")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error improving model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_improved_model()

