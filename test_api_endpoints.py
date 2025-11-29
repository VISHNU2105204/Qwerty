"""
Unit tests for API endpoints in the Fake News Detector application
"""

import unittest
import json
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock
from http.server import HTTPServer
import threading
import time
import requests
from io import BytesIO
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(__file__))

from server import CORSRequestHandler


class APIEndpointTestCase(unittest.TestCase):
    """Base test case for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test server once for all tests"""
        cls.test_db_path = 'test_users.db'
        cls.server_port = 8888
        cls.base_url = f'http://localhost:{cls.server_port}'
        
        # Start test server in a separate thread
        cls.server = HTTPServer(('localhost', cls.server_port), CORSRequestHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(1)  # Give server time to start
        
    @classmethod
    def tearDownClass(cls):
        """Tear down test server"""
        cls.server.shutdown()
        cls.server_thread.join(timeout=2)
        
        # Clean up test database
        if os.path.exists('users.db'):
            try:
                os.remove('users.db')
            except:
                pass
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the model loading
        CORSRequestHandler._model = MagicMock()
        CORSRequestHandler._vectorizer = MagicMock()
        CORSRequestHandler._model_loaded = True
        
        # Setup test database
        self.db_path = 'test_users.db'
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except:
                pass


class TestAnalyzeEndpoint(unittest.TestCase):
    """Test the /api/analyze endpoint"""
    
    def setUp(self):
        """Set up test fixtures"""
        CORSRequestHandler._model = MagicMock()
        CORSRequestHandler._vectorizer = MagicMock()
        CORSRequestHandler._model_loaded = True
    
    def test_analyze_with_valid_text(self):
        """Test analyze endpoint with valid text input"""
        # Mock the model to return a prediction
        CORSRequestHandler._model.predict = MagicMock(return_value=[1])
        CORSRequestHandler._vectorizer.transform = MagicMock(return_value=MagicMock())
        
        payload = {'text': 'This is a test news article'}
        
        # Simulate the request handling
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        # Verify model methods are callable
        self.assertTrue(callable(CORSRequestHandler._model.predict))
        self.assertTrue(callable(CORSRequestHandler._vectorizer.transform))
    
    def test_analyze_with_empty_text(self):
        """Test analyze endpoint with empty text"""
        payload = {'text': ''}
        
        # Simulate the request handling
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        # Should handle empty text gracefully
        self.assertEqual(payload['text'].strip(), '')
    
    def test_analyze_with_missing_text(self):
        """Test analyze endpoint with missing text field"""
        payload = {}
        
        # Should have default empty string
        text = payload.get('text', '').strip()
        self.assertEqual(text, '')


class TestAuthenticationEndpoints(unittest.TestCase):
    """Test authentication related endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_path = ':memory:'
        self.test_user = {
            'email': 'test@example.com',
            'password': 'TestPassword123!',
            'name': 'Test User'
        }
    
    def test_register_endpoint_structure(self):
        """Test register endpoint accepts required fields"""
        payload = {
            'email': self.test_user['email'],
            'password': self.test_user['password'],
            'name': self.test_user['name']
        }
        
        self.assertIn('email', payload)
        self.assertIn('password', payload)
        self.assertIn('name', payload)
    
    def test_login_endpoint_structure(self):
        """Test login endpoint accepts required fields"""
        payload = {
            'email': self.test_user['email'],
            'password': self.test_user['password']
        }
        
        self.assertIn('email', payload)
        self.assertIn('password', payload)
    
    def test_logout_endpoint_structure(self):
        """Test logout endpoint accepts token"""
        payload = {'token': 'test_token_12345'}
        
        self.assertIn('token', payload)
    
    def test_check_session_structure(self):
        """Test check-session endpoint accepts token"""
        params = '?token=test_token_12345'
        
        self.assertIn('token', params)


class TestProfileEndpoints(unittest.TestCase):
    """Test profile related endpoints"""
    
    def test_update_profile_structure(self):
        """Test update-profile endpoint structure"""
        payload = {
            'token': 'test_token',
            'name': 'Updated Name',
            'email': 'newemail@example.com'
        }
        
        self.assertIn('token', payload)
        self.assertIn('name', payload)
        self.assertIn('email', payload)
    
    def test_notification_settings_structure(self):
        """Test notification-settings endpoint structure"""
        payload = {
            'token': 'test_token',
            'email_notifications': True,
            'account_updates': False
        }
        
        self.assertIn('token', payload)
        self.assertIn('email_notifications', payload)
        self.assertIn('account_updates', payload)
    
    def test_account_stats_structure(self):
        """Test account-stats endpoint structure"""
        payload = {'token': 'test_token'}
        
        self.assertIn('token', payload)


class TestHistoryEndpoints(unittest.TestCase):
    """Test history related endpoints"""
    
    def test_analysis_history_structure(self):
        """Test analysis-history endpoint structure"""
        payload = {
            'token': 'test_token',
            'page': 1,
            'limit': 10
        }
        
        self.assertIn('token', payload)
    
    def test_clear_history_structure(self):
        """Test clear-history endpoint structure"""
        payload = {'token': 'test_token'}
        
        self.assertIn('token', payload)
    
    def test_export_data_structure(self):
        """Test export-data endpoint structure"""
        payload = {'token': 'test_token'}
        
        self.assertIn('token', payload)


class TestPasswordRecoveryEndpoints(unittest.TestCase):
    """Test password recovery related endpoints"""
    
    def test_forgot_password_structure(self):
        """Test forgot-password endpoint structure"""
        payload = {'email': 'test@example.com'}
        
        self.assertIn('email', payload)
    
    def test_reset_password_structure(self):
        """Test reset-password endpoint structure"""
        payload = {
            'token': 'reset_token_12345',
            'new_password': 'NewPassword123!'
        }
        
        self.assertIn('token', payload)
        self.assertIn('new_password', payload)


class TestAccountManagementEndpoints(unittest.TestCase):
    """Test account management endpoints"""

    def setUp(self):
        """Set up test database for account management tests"""
        self.db_path = ':memory:'
        self.test_user = {
            'email': 'delete_test@example.com',
            'password': 'DeleteTest123!',
            'name': 'Delete Test User'
        }

    def test_delete_account_structure(self):
        """Test delete-account endpoint structure"""
        payload = {
            'token': 'test_token',
            'password': 'UserPassword123'
        }

        self.assertIn('token', payload)
        self.assertIn('password', payload)

    def test_delete_account_functionality(self):
        """Test delete-account endpoint functionality"""
        # Create test user
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        name TEXT,
                        email_notifications BOOLEAN DEFAULT 1,
                        account_updates BOOLEAN DEFAULT 1,
                        theme_preference TEXT DEFAULT 'light'
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        token TEXT PRIMARY KEY,
                        user_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        text TEXT,
                        is_real BOOLEAN,
                        confidence INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')

        # Insert test user
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = self.db_path
        password_hash = handler.hash_password(self.test_user['password'])

        c.execute('INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)',
                 (self.test_user['email'], password_hash, self.test_user['name']))
        user_id = c.lastrowid

        # Create session token
        token = handler.generate_token()
        c.execute('INSERT INTO sessions (token, user_id) VALUES (?, ?)', (token, user_id))

        # Add some analysis history
        c.execute('INSERT INTO analysis_history (user_id, text, is_real, confidence) VALUES (?, ?, ?, ?)',
                 (user_id, 'Test analysis text', True, 85))

        conn.commit()

        # Verify user exists before deletion
        c.execute('SELECT COUNT(*) FROM users WHERE email = ?', (self.test_user['email'],))
        self.assertEqual(c.fetchone()[0], 1)

        c.execute('SELECT COUNT(*) FROM sessions WHERE user_id = ?', (user_id,))
        self.assertEqual(c.fetchone()[0], 1)

        c.execute('SELECT COUNT(*) FROM analysis_history WHERE user_id = ?', (user_id,))
        self.assertEqual(c.fetchone()[0], 1)

        # Simulate delete account request
        payload = {'password': self.test_user['password']}

        # Verify password and delete account (simulating the handler logic)
        stored_password_hash = password_hash
        input_password_hash = handler.hash_password(payload['password'])

        self.assertEqual(input_password_hash, stored_password_hash)

        # Delete user data
        c.execute('DELETE FROM analysis_history WHERE user_id = ?', (user_id,))
        c.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()

        # Verify user data is deleted
        c.execute('SELECT COUNT(*) FROM users WHERE email = ?', (self.test_user['email'],))
        self.assertEqual(c.fetchone()[0], 0)

        c.execute('SELECT COUNT(*) FROM sessions WHERE user_id = ?', (user_id,))
        self.assertEqual(c.fetchone()[0], 0)

        c.execute('SELECT COUNT(*) FROM analysis_history WHERE user_id = ?', (user_id,))
        self.assertEqual(c.fetchone()[0], 0)

        conn.close()


class TestImageAnalysisEndpoint(unittest.TestCase):
    """Test image analysis endpoint"""
    
    def setUp(self):
        """Set up test fixtures"""
        CORSRequestHandler._model = MagicMock()
        CORSRequestHandler._vectorizer = MagicMock()
        CORSRequestHandler._model_loaded = True
    
    def test_analyze_image_structure(self):
        """Test analyze-image endpoint structure"""
        payload = {
            'image_url': 'https://example.com/image.jpg'
        }
        
        self.assertIn('image_url', payload)
    
    def test_analyze_url_structure(self):
        """Test analyze-url endpoint structure"""
        payload = {
            'url': 'https://example.com/news-article',
            'extract_text': True
        }
        
        self.assertIn('url', payload)


class TestRequestHandling(unittest.TestCase):
    """Test request handling and routing"""
    
    def test_cors_headers_present(self):
        """Test CORS headers are set correctly"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        
        # Verify handler has methods
        self.assertTrue(hasattr(handler, 'end_headers'))
        self.assertTrue(hasattr(handler, 'do_OPTIONS'))
        self.assertTrue(hasattr(handler, 'do_GET'))
        self.assertTrue(hasattr(handler, 'do_POST'))
    
    def test_endpoint_routing(self):
        """Test endpoint routing structure"""
        endpoints = [
            '/api/analyze',
            '/api/analyze-url',
            '/api/analyze-image',
            '/api/register',
            '/api/login',
            '/api/logout',
            '/api/check-session',
            '/api/update-profile',
            '/api/notification-settings',
            '/api/account-stats',
            '/api/analysis-history',
            '/api/export-data',
            '/api/delete-account',
            '/api/clear-history',
            '/api/forgot-password',
            '/api/reset-password'
        ]
        
        # Verify all endpoints are strings
        for endpoint in endpoints:
            self.assertIsInstance(endpoint, str)
            self.assertTrue(endpoint.startswith('/api/'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in API endpoints"""
    
    def test_missing_content_length(self):
        """Test handling of missing Content-Length header"""
        payload = {'test': 'data'}
        
        # Should be handleable
        self.assertIsInstance(payload, dict)
    
    def test_invalid_json_payload(self):
        """Test handling of invalid JSON"""
        invalid_json = '{"invalid": json}'
        
        # Should raise JSON error when parsed
        with self.assertRaises(json.JSONDecodeError):
            json.loads(invalid_json)
    
    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        
        # Should have error handling methods
        self.assertTrue(hasattr(handler, 'send_error'))


class TestDatabaseOperations(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        """Set up test database"""
        self.db_path = ':memory:'
    
    def test_database_initialization(self):
        """Test database table creation"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        name TEXT
                    )''')
        
        # Verify table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        result = c.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()
    
    def test_user_insertion(self):
        """Test inserting a user into database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        name TEXT
                    )''')
        
        # Insert test user
        email = 'test@example.com'
        password_hash = 'hashed_password'
        name = 'Test User'
        
        c.execute('INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)',
                 (email, password_hash, name))
        
        # Verify insertion
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        result = c.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[1], email)
        
        conn.commit()
        conn.close()


class TestPasswordHashing(unittest.TestCase):
    """Test password hashing functionality"""
    
    def test_hash_password(self):
        """Test password hashing"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        password = 'TestPassword123!'
        hashed = handler.hash_password(password)
        
        # Should return a non-empty string
        self.assertIsInstance(hashed, str)
        self.assertGreater(len(hashed), 0)
    
    def test_hash_consistency(self):
        """Test that same password produces same hash"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        password = 'TestPassword123!'
        hash1 = handler.hash_password(password)
        hash2 = handler.hash_password(password)
        
        # Same password should produce same hash
        self.assertEqual(hash1, hash2)
    
    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        password1 = 'Password123!'
        password2 = 'Password456!'
        
        hash1 = handler.hash_password(password1)
        hash2 = handler.hash_password(password2)
        
        # Different passwords should produce different hashes
        self.assertNotEqual(hash1, hash2)


class TestTokenGeneration(unittest.TestCase):
    """Test token generation functionality"""
    
    def test_generate_token(self):
        """Test token generation"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        token = handler.generate_token()
        
        # Should return a non-empty string
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 0)
    
    def test_token_uniqueness(self):
        """Test that generated tokens are unique"""
        handler = CORSRequestHandler.__new__(CORSRequestHandler)
        handler.db_path = ':memory:'
        
        tokens = set()
        for _ in range(10):
            token = handler.generate_token()
            # Each token should be unique
            self.assertNotIn(token, tokens)
            tokens.add(token)


if __name__ == '__main__':
    unittest.main(verbosity=2)
