@echo off
title Fake News Detection Server
color 0A
echo ========================================
echo   Fake News Detection Server
echo ========================================
echo.
echo Starting server on http://127.0.0.1:5500
echo Press Ctrl+C to stop the server
echo.
python server.py
if errorlevel 1 (
    echo.
    echo Error: Server failed to start!
    pause
)

