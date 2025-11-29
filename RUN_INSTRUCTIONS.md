# How to Run the Fake News Detection Project

## Prerequisites

1. **Install Python 3.7 or higher**
   - Download from https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Install Tesseract OCR** (required for image analysis)
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location (usually `C:\Program Files\Tesseract-OCR`)
   - Add to PATH or update `pytesseract.pytesseract.tesseract_cmd` in the code if needed

## Installation Steps

1. **Navigate to the project directory:**
   ```bash
   cd "C:\Users\THARUN\Downloads\Final_Year_PROJECT-main\Final_Year_PROJECT-main"
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Run the Full Server (Recommended)
This includes user authentication, analysis history, and all features:
```bash
python server.py
```
The server will start at: **http://127.0.0.1:5500**

### Option 2: Run the Simple Server
This is a simpler version without authentication:
```bash
python working_server.py
```
The server will start at: **http://127.0.0.1:5500**

### Option 3: Run Streamlit App
For a different interface using Streamlit:
```bash
streamlit run app.py
```

## Accessing the Application

1. Open your web browser
2. Navigate to: `http://127.0.0.1:5500`
3. Open `index.html` or `simple_detect.html` to use the application

## Features

- **Text Analysis**: Analyze news articles by pasting text
- **URL Analysis**: Analyze news from URLs
- **Image Analysis**: Extract and analyze text from images using OCR
- **User Authentication**: Sign up, login, and track analysis history
- **Analysis History**: View past analyses and statistics

## Troubleshooting

- If Python is not found, make sure it's installed and added to PATH
- If Tesseract OCR errors occur, ensure Tesseract is installed and in PATH
- If model files (`lr_model.jb`, `vectorizer.jb`) are missing, you may need to train the model first using `retrain_model.py`

