import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

def check_model_accuracy():
    """Check the model's accuracy on the training dataset"""
    print("Checking Model Accuracy...")
    print("=" * 50)
    
    try:
        # Load model and vectorizer
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        print("✅ Model loaded successfully")
        
        # Load dataset
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
        
        # Add labels
        true_df['label'] = 1  # Real news
        fake_df['label'] = 0   # Fake news
        
        # Combine and shuffle
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Split data
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Transform test data
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 MODEL PERFORMANCE METRICS:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print(f"\n📋 DETAILED REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 CONFUSION MATRIX:")
        print("                 Predicted")
        print("               Fake  Real")
        print(f"Actual Fake    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Real    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Calculate precision and recall
        fake_precision = cm[0,0] / (cm[0,0] + cm[1,0])
        fake_recall = cm[0,0] / (cm[0,0] + cm[0,1])
        real_precision = cm[1,1] / (cm[1,1] + cm[0,1])
        real_recall = cm[1,1] / (cm[1,1] + cm[1,0])
        
        print(f"\n🎯 PRECISION & RECALL:")
        print(f"Fake News - Precision: {fake_precision:.3f}, Recall: {fake_recall:.3f}")
        print(f"Real News - Precision: {real_precision:.3f}, Recall: {real_recall:.3f}")
        
        # Overall assessment
        if accuracy >= 0.95:
            print(f"\n🏆 EXCELLENT: Model accuracy is {accuracy*100:.1f}% - Very reliable!")
        elif accuracy >= 0.90:
            print(f"\n✅ GOOD: Model accuracy is {accuracy*100:.1f}% - Reliable for most cases")
        elif accuracy >= 0.80:
            print(f"\n⚠️  FAIR: Model accuracy is {accuracy*100:.1f}% - May need improvement")
        else:
            print(f"\n❌ POOR: Model accuracy is {accuracy*100:.1f}% - Needs retraining")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return None

if __name__ == "__main__":
    check_model_accuracy()
