from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.request
import urllib.parse
import threading
import time
import joblib
import os
import re
import base64
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import uuid
import hashlib
import sqlite3
import os

# Set the document root to the current directory
DOCUMENT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Database setup
DB_NAME = 'users.db'

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            expires_at REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            token TEXT UNIQUE,
            otp TEXT,
            expiry INTEGER,
            used BOOLEAN DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text TEXT,
            url TEXT,
            is_real BOOLEAN,
            confidence INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Remove directory from kwargs if present and pass it to super()
        directory = kwargs.pop('directory', DOCUMENT_ROOT)
        super().__init__(*args, directory=directory, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Handle check-session as GET request
        if self.path == '/api/check-session':
            self.handle_check_session()
        else:
            # Default behavior for static files
            super().do_GET()

    def do_POST(self):
        print(f"POST request to: {self.path}")
        
        if self.path == '/api/analyze':
            self.handle_analyze()
        elif self.path == '/api/analyze-url':
            self.handle_analyze_url()
        elif self.path == '/api/register':
            self.handle_register()
        elif self.path == '/api/login':
            self.handle_login()
        elif self.path == '/api/check-session':
            self.handle_check_session()
        elif self.path == '/api/forgot-password':
            self.handle_forgot_password()
        elif self.path == '/api/verify-otp':
            self.handle_verify_otp()
        elif self.path == '/api/reset-password':
            self.handle_reset_password()
        elif self.path == '/api/delete-account':
            self.handle_delete_account()
        elif self.path == '/api/analysis-history':
            self.handle_get_analysis_history()
        else:
            print(f"404 - Path not found: {self.path}")
            self.send_error(404)

    def handle_analyze(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            
            # Check for authorization header to associate with user
            user_id = None
            auth_header = self.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header.replace('Bearer ', '')
                # Get user_id from session
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('SELECT user_id FROM sessions WHERE token = ?', (token,))
                session_row = cursor.fetchone()
                if session_row:
                    user_id = session_row[0]
                conn.close()
            
            if not text.strip():
                result = {
                    'isReal': False,
                    'confidence': 0,
                    'message': 'Please enter some text to analyze'
                }
            else:
                # Load the ML model and vectorizer
                try:
                    vectorizer = joblib.load("vectorizer.jb")
                    model = joblib.load("lr_model.jb")
                    
                    # Transform the input text
                    transform_input = vectorizer.transform([text])
                    prediction = model.predict(transform_input)
                    
                    # Get prediction probability for confidence
                    try:
                        prediction_proba = model.predict_proba(transform_input)
                        confidence = int(max(prediction_proba[0]) * 100)
                    except:
                        confidence = 85  # Default confidence if predict_proba not available
                    
                    is_real = prediction[0] == 1
                    
                    result = {
                        'isReal': bool(is_real),
                        'confidence': confidence,
                        'message': 'Real news' if is_real else 'Fake news'
                    }
                    
                    # Save to analysis history if user is logged in
                    if user_id:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO analysis_history (user_id, text, is_real, confidence)
                            VALUES (?, ?, ?, ?)
                        ''', (user_id, text, is_real, confidence))
                        conn.commit()
                        conn.close()
                    
                except FileNotFoundError:
                    # Fallback to mock analysis if model files not found
                    import random
                    is_real = random.choice([True, False])
                    confidence = random.randint(70, 95)
                    
                    result = {
                        'isReal': is_real,
                        'confidence': confidence,
                        'message': 'Real news' if is_real else 'Fake news'
                    }
                    
                    # Save to analysis history if user is logged in
                    if user_id:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO analysis_history (user_id, text, is_real, confidence)
                            VALUES (?, ?, ?, ?)
                        ''', (user_id, text, is_real, confidence))
                        conn.commit()
                        conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            print(f"Error in handle_analyze: {e}")
            self.send_error(500, str(e))

    def handle_analyze_url(self):
        print("Handling URL analysis request")
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            url = data.get('url', '')
            print(f"Analyzing URL: {url}")
            
            # Check for authorization header to associate with user
            user_id = None
            auth_header = self.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header.replace('Bearer ', '')
                # Get user_id from session
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('SELECT user_id FROM sessions WHERE token = ?', (token,))
                session_row = cursor.fetchone()
                if session_row:
                    user_id = session_row[0]
                conn.close()
            
            if not url.strip():
                result = {
                    'isReal': False,
                    'confidence': 0,
                    'message': 'Please provide a valid URL'
                }
            else:
                # Analyze URL
                result = self.analyze_url_content(url)
                
                # Save to analysis history if user is logged in
                if user_id and result.get('isReal') is not None:
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO analysis_history (user_id, url, is_real, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, url, result['isReal'], result['confidence']))
                    conn.commit()
                    conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            print(f"Error in handle_analyze_url: {e}")
            self.send_error(500, str(e))

    def analyze_url_content(self, url):
        """Analyze content from URL"""
        try:
            print(f"Fetching URL: {url}")
            
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
            
            print(f"URL fetched successfully, status: {response.status_code}")
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_content = ""
            
            # Try to find article content
            article = soup.find('article')
            if article:
                text_content = article.get_text()
                print("Found article content")
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
                        print(f"Found content with selector: {selector}")
                        break
            
            # If no specific content found, get all text
            if not text_content:
                text_content = soup.get_text()
                print("Using all page text")
            
            # Clean text
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            print(f"Extracted text length: {len(text_content)}")
            
            if len(text_content) < 50:
                return {"isReal": False, "confidence": 0, "message": "Insufficient content extracted from URL"}
            
            # Analyze the extracted text using existing model
            return self.analyze_text_with_model(text_content)
            
        except requests.exceptions.RequestException as e:
            print(f"URL request error: {e}")
            return {"isReal": False, "confidence": 0, "message": f"URL access error: {str(e)}"}
        except Exception as e:
            print(f"URL analysis error: {e}")
            return {"isReal": False, "confidence": 0, "message": f"URL analysis error: {str(e)}"}

    def analyze_text_with_model(self, text):
        """Analyze text using the trained model"""
        try:
            vectorizer = joblib.load("vectorizer.jb")
            model = joblib.load("lr_model.jb")
            
            text_tfidf = vectorizer.transform([text])
            prediction = model.predict(text_tfidf)[0]
            confidence = model.predict_proba(text_tfidf)[0].max() * 100
            
            return {
                "isReal": bool(prediction),
                "confidence": int(confidence),
                "message": "Real news" if prediction else "Fake news"
            }
        except Exception as e:
            print(f"Model analysis error: {e}")
            return {"isReal": False, "confidence": 0, "message": f"Model analysis error: {str(e)}"}

    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def generate_token(self):
        """Generate a unique token"""
        return str(uuid.uuid4())

    def handle_register(self):
        """Handle user registration"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            name = data.get('name', '').strip()
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            print(f"Registration attempt for: {email}")
            
            # Validate input
            if not name or not email or not password:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'All fields are required'
                }).encode())
                return
            
            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT email FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                conn.close()
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Email already registered'
                }).encode())
                return
            
            # Create new user
            user_id = str(uuid.uuid4())
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (user_id, name, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (user_id, name, email, password_hash))
            conn.commit()
            conn.close()
            
            # Generate token
            token = self.generate_token()
            expires_at = time.time() + 86400 * 30  # 30 days
            
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (token, user_id, email, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (token, user_id, email, expires_at))
            conn.commit()
            conn.close()
            
            print(f"User registered successfully: {email}")
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'token': token,
                'user': {
                    'user_id': user_id,
                    'name': name,
                    'email': email
                }
            }).encode())
            
        except Exception as e:
            print(f"Error in handle_register: {e}")
            self.send_error(500, str(e))

    def handle_login(self):
        """Handle user login"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            print(f"Login attempt for: {email}")
            
            # Validate input
            if not email or not password:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Email and password are required'
                }).encode())
                return
            
            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT user_id, name, email, password_hash FROM users WHERE email = ?', (email,))
            user_row = cursor.fetchone()
            
            if not user_row:
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Invalid email or password'
                }).encode())
                return
            
            # Verify password
            user_id, name, email, stored_password_hash = user_row
            password_hash = self.hash_password(password)
            
            if stored_password_hash != password_hash:
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Invalid email or password'
                }).encode())
                return
            
            # Generate token
            token = self.generate_token()
            expires_at = time.time() + 86400 * 30  # 30 days
            
            cursor.execute('''
                INSERT INTO sessions (token, user_id, email, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (token, user_id, email, expires_at))
            conn.commit()
            conn.close()
            
            print(f"User logged in successfully: {email}")
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'token': token,
                'user': {
                    'user_id': user_id,
                    'name': name,
                    'email': email
                }
            }).encode())
            
        except Exception as e:
            print(f"Error in handle_login: {e}")
            self.send_error(500, str(e))

    def handle_check_session(self):
        """Check if session is valid"""
        try:
            # Get authorization header
            auth_header = self.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Invalid authorization header'
                }).encode())
                return
            
            token = auth_header.replace('Bearer ', '')
            
            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Check if token exists and is valid
            cursor.execute('''
                SELECT s.user_id, s.email, s.expires_at, u.name
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.token = ?
            ''', (token,))
            session_row = cursor.fetchone()
            
            if not session_row:
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Invalid token'
                }).encode())
                return
            
            user_id, email, expires_at, name = session_row
            
            # Check if token is expired
            if expires_at < time.time():
                cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
                conn.commit()
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Token expired'
                }).encode())
                return
            
            # Send success response
            conn.close()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'user': {
                    'user_id': user_id,
                    'name': name,
                    'email': email
                }
            }).encode())
            
        except Exception as e:
            print(f"Error in handle_check_session: {e}")
            self.send_error(500, str(e))

    def handle_forgot_password(self):
        """Handle forgot password request - generate and send OTP"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            email = data.get('email', '').strip().lower()

            if not email:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Email is required'}).encode())
                return

            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute('SELECT user_id, name FROM users WHERE email = ?', (email,))
            user_row = cursor.fetchone()

            if not user_row:
                # For development, generate a dummy OTP even for non-existent users
                import secrets
                dummy_otp = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
                
                # Store dummy OTP with expiry (10 minutes from now)
                import time
                import uuid
                expiry = int(time.time()) + 600  # 10 minutes
                reset_token = str(uuid.uuid4())
                
                # Insert dummy OTP with null user_id for non-existent users
                cursor.execute('INSERT INTO password_resets (user_id, token, otp, expiry) VALUES (?, ?, ?, ?)',
                             (None, reset_token, dummy_otp, expiry))
                conn.commit()
                conn.close()
                
                # Don't reveal if email exists or not for security, but include OTP for development
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'message': 'If an account exists for this email, we\'ve sent an OTP.',
                    'otp': dummy_otp  # For development/testing
                }).encode())
                return

            user_id, name = user_row

            # Generate 6-digit OTP
            import secrets
            otp = ''.join([str(secrets.randbelow(10)) for _ in range(6)])

            # Store OTP with expiry (10 minutes from now)
            import time
            expiry = int(time.time()) + 600  # 10 minutes

            # Create table for password resets if it doesn't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS password_resets (
                                id INTEGER PRIMARY KEY,
                                user_id TEXT,
                                token TEXT UNIQUE,
                                otp TEXT,
                                expiry INTEGER,
                                used BOOLEAN DEFAULT 0
                            )''')

            # Remove any existing reset tokens/OTPs for this user
            cursor.execute('DELETE FROM password_resets WHERE user_id = ?', (user_id,))

            # Insert new OTP
            import uuid
            reset_token = str(uuid.uuid4())
            cursor.execute('INSERT INTO password_resets (user_id, token, otp, expiry) VALUES (?, ?, ?, ?)',
                         (user_id, reset_token, otp, expiry))
            conn.commit()
            conn.close()

            # For development, include OTP in response
            # In production, you would send it via email
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True,
                'message': 'OTP has been generated. Check server console for OTP details.',
                'otp': otp  # For development/testing - will show in console
            }).encode())

        except Exception as e:
            print(f"Error in handle_forgot_password: {e}")
            self.send_error(500, str(e))

    def handle_verify_otp(self):
        """Verify OTP and return reset token for password reset"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            email = data.get('email', '').strip().lower()
            otp = data.get('otp', '').strip()

            if not email or not otp:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Email and OTP are required'}).encode())
                return

            # Verify OTP
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            import time
            current_time = int(time.time())

            # Find valid OTP for this email
            # First check if user exists
            cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
            user_row = cursor.fetchone()
            
            if user_row:
                # User exists, verify OTP with user join
                user_id = user_row[0]
                cursor.execute('''
                    SELECT pr.token, pr.otp, pr.expiry, pr.used 
                    FROM password_resets pr
                    WHERE pr.user_id = ? AND pr.otp = ? AND pr.expiry > ? AND pr.used = 0
                ''', (user_id, otp, current_time))
            else:
                # User doesn't exist, verify OTP by email only (for development)
                # Look for OTP with NULL user_id (dummy OTPs)
                cursor.execute('''
                    SELECT pr.token, pr.otp, pr.expiry, pr.used 
                    FROM password_resets pr
                    WHERE pr.otp = ? AND pr.user_id IS NULL AND pr.expiry > ? AND pr.used = 0
                ''', (otp, current_time))

            reset_record = cursor.fetchone()

            if reset_record:
                reset_token, stored_otp, expiry, used = reset_record
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'message': 'OTP verified successfully',
                    'reset_token': reset_token
                }).encode())
            else:
                conn.close()
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Invalid or expired OTP. Please request a new one.'
                }).encode())

        except Exception as e:
            print(f"Error in handle_verify_otp: {e}")
            self.send_error(500, str(e))

    def handle_reset_password(self):
        """Reset user password"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            token = data.get('token', '').strip()
            new_password = data.get('newPassword', '').strip()

            if not token or not new_password:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Token and new password are required'}).encode())
                return

            # Validate password strength (basic)
            if len(new_password) < 6:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Password must be at least 6 characters long'}).encode())
                return

            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            # Check if reset token exists and is valid
            import time
            current_time = int(time.time())

            cursor.execute('''
                SELECT pr.user_id, pr.used FROM password_resets pr
                WHERE pr.token = ? AND pr.expiry > ? AND pr.used = 0
            ''', (token, current_time))

            reset_record = cursor.fetchone()

            if not reset_record:
                conn.close()
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid or expired reset token'}).encode())
                return

            user_id, used = reset_record

            # Update password
            new_password_hash = self.hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE user_id = ?', (new_password_hash, user_id))

            # Mark reset token as used
            cursor.execute('UPDATE password_resets SET used = 1 WHERE token = ?', (token,))

            conn.commit()
            conn.close()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'message': 'Password reset successfully. You can now login with your new password.',
                'redirect': 'login.html'
            }).encode())

        except Exception as e:
            self.send_error(500, str(e))

    def handle_delete_account(self):
        """Delete user account and all associated data"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            # Get authorization header
            auth_header = self.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid authorization header'}).encode())
                return

            token = auth_header.replace('Bearer ', '')

            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            # Check if token exists and is valid
            cursor.execute('''
                SELECT s.user_id, s.email, s.expires_at, u.name, u.password_hash
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.token = ?
            ''', (token,))
            session_row = cursor.fetchone()

            if not session_row:
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid token'}).encode())
                return

            user_id, email, expires_at, name, stored_password_hash = session_row

            # Check if token is expired
            if expires_at < time.time():
                cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
                conn.commit()
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Token expired'}).encode())
                return

            # Verify password if provided
            password = data.get('password', '')
            if password:
                input_password_hash = self.hash_password(password)
                if stored_password_hash != input_password_hash:
                    conn.close()
                    self.send_response(401)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid password'}).encode())
                    return

            # Delete user data from all tables
            cursor.execute('DELETE FROM analysis_history WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM password_resets WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            
            conn.commit()
            conn.close()

            print(f"User account deleted successfully: {email}")

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'message': 'Account deleted successfully',
                'redirect': 'index.html'
            }).encode())

        except Exception as e:
            print(f"Error in handle_delete_account: {e}")
            self.send_error(500, str(e))

    def handle_get_analysis_history(self):
        """Get user's analysis history"""
        try:
            # Get authorization header
            auth_header = self.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid authorization header'}).encode())
                return

            token = auth_header.replace('Bearer ', '')

            # Connect to database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            # Check if token exists and is valid
            cursor.execute('''
                SELECT s.user_id, s.expires_at
                FROM sessions s
                WHERE s.token = ?
            ''', (token,))
            session_row = cursor.fetchone()

            if not session_row:
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid token'}).encode())
                return

            user_id, expires_at = session_row

            # Check if token is expired
            if expires_at < time.time():
                cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
                conn.commit()
                conn.close()
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Token expired'}).encode())
                return

            # Get user's analysis history
            cursor.execute('''
                SELECT id, text, url, is_real, confidence, created_at
                FROM analysis_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            ''', (user_id,))
            
            history_rows = cursor.fetchall()
            history = []
            
            for row in history_rows:
                history.append({
                    'id': row[0],
                    'text': row[1][:100] + '...' if row[1] and len(row[1]) > 100 else row[1],
                    'url': row[2],
                    'isReal': bool(row[3]),
                    'confidence': row[4],
                    'createdAt': row[5]
                })
            
            conn.close()

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'history': history
            }).encode())

        except Exception as e:
            print(f"Error in handle_get_analysis_history: {e}")
            self.send_error(500, str(e))

def start_server():
    server = HTTPServer(('127.0.0.1', 5500), CORSRequestHandler)
    print("Working server running at http://127.0.0.1:5500")
    print("Enhanced with URL analysis support!")
    print("Server started successfully!")
    server.serve_forever()

if __name__ == '__main__':
    start_server()
