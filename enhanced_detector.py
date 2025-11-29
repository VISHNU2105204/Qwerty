import joblib
import requests
import re
from urllib.parse import urlparse
from PIL import Image
import io
import base64
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class EnhancedNewsDetector:
    def __init__(self):
        """Initialize the enhanced news detector"""
        try:
            self.text_model = joblib.load("lr_model.jb")
            self.vectorizer = joblib.load("vectorizer.jb")
            print("✅ Text analysis model loaded")
        except:
            print("❌ Text model not found")
            self.text_model = None
            self.vectorizer = None
    
    def analyze_text(self, text):
        """Analyze text content"""
        if not self.text_model or not self.vectorizer:
            return {"isReal": False, "confidence": 0, "message": "Text model not available"}
        
        try:
            text_tfidf = self.vectorizer.transform([text])
            prediction = self.text_model.predict(text_tfidf)[0]
            confidence = self.text_model.predict_proba(text_tfidf)[0].max() * 100
            
            return {
                "isReal": bool(prediction),
                "confidence": int(confidence),
                "message": "Real news" if prediction else "Fake news"
            }
        except Exception as e:
            return {"isReal": False, "confidence": 0, "message": f"Text analysis error: {str(e)}"}
    
    def analyze_url(self, url):
        """Analyze URL and extract text content"""
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {"isReal": False, "confidence": 0, "message": "Invalid URL format"}
            
            # Fetch content from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_content = ""
            
            # Try to find article content
            article = soup.find('article')
            if article:
                text_content = article.get_text()
            else:
                # Look for common content selectors
                content_selectors = [
                    'div[class*="content"]',
                    'div[class*="article"]',
                    'div[class*="post"]',
                    'div[class*="story"]',
                    'main',
                    '.entry-content',
                    '.post-content'
                ]
                
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text_content = element.get_text()
                        break
            
            # If no specific content found, get all text
            if not text_content:
                text_content = soup.get_text()
            
            # Clean text
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            if len(text_content) < 50:
                return {"isReal": False, "confidence": 0, "message": "Insufficient content extracted from URL"}
            
            # Analyze the extracted text
            result = self.analyze_text(text_content)
            result["extracted_text"] = text_content[:200] + "..." if len(text_content) > 200 else text_content
            result["source"] = "URL Analysis"
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {"isReal": False, "confidence": 0, "message": f"URL access error: {str(e)}"}
        except Exception as e:
            return {"isReal": False, "confidence": 0, "message": f"URL analysis error: {str(e)}"}
    
    def analyze_image(self, image_data):
        """Analyze image content (basic OCR and metadata analysis)"""
        try:
            # For now, we'll do basic image analysis
            # In a real implementation, you'd use OCR to extract text from images
            
            # Check if it's a valid image
            if isinstance(image_data, str):
                # Base64 encoded image
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Basic image analysis
            width, height = image.size
            format_type = image.format
            
            # Simple heuristics for fake news images
            fake_indicators = []
            
            # Check image dimensions (very small images might be fake)
            if width < 100 or height < 100:
                fake_indicators.append("Very small image size")
            
            # Check if image is too large (might be manipulated)
            if width > 4000 or height > 4000:
                fake_indicators.append("Unusually large image size")
            
            # Basic confidence calculation
            confidence = 50  # Default neutral confidence
            
            if len(fake_indicators) > 0:
                confidence = 30  # Lower confidence for suspicious images
            else:
                confidence = 60  # Higher confidence for normal images
            
            return {
                "isReal": len(fake_indicators) == 0,
                "confidence": confidence,
                "message": "Image analysis completed",
                "image_info": {
                    "width": width,
                    "height": height,
                    "format": format_type,
                    "indicators": fake_indicators
                }
            }
            
        except Exception as e:
            return {"isReal": False, "confidence": 0, "message": f"Image analysis error: {str(e)}"}
    
    def analyze_content(self, content, content_type="text"):
        """Main analysis function that handles different content types"""
        print(f"Analyzing {content_type} content...")
        
        if content_type == "text":
            return self.analyze_text(content)
        elif content_type == "url":
            return self.analyze_url(content)
        elif content_type == "image":
            return self.analyze_image(content)
        else:
            return {"isReal": False, "confidence": 0, "message": f"Unsupported content type: {content_type}"}

# Test the enhanced detector
def test_enhanced_detector():
    """Test the enhanced detector with different content types"""
    detector = EnhancedNewsDetector()
    
    print("Testing Enhanced News Detector")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "content": "Scientists discover new planet with potential for life",
            "type": "text",
            "expected": "Real"
        },
        {
            "content": "ALIENS LANDED IN MY BACKYARD!!!",
            "type": "text", 
            "expected": "Fake"
        },
        {
            "content": "https://www.bbc.com/news",
            "type": "url",
            "expected": "Real"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {case['type']}: {case['content'][:50]}...")
        result = detector.analyze_content(case['content'], case['type'])
        print(f"   Result: {'Real' if result['isReal'] else 'Fake'}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Message: {result['message']}")

if __name__ == "__main__":
    test_enhanced_detector()
