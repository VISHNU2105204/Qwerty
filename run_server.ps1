# PowerShell script to run the server
Write-Host "Starting Fake News Detection Server..." -ForegroundColor Green
Write-Host ""

# Use the Python that's available in the system PATH instead of hardcoded path
python server.py