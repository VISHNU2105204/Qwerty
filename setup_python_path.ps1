# Script to add Anaconda Python to PATH permanently
# Run this script as Administrator for permanent changes, or it will only affect current session

$anacondaPath = "C:\Users\THARUN\anaconda3"
$anacondaScripts = "$anacondaPath\Scripts"

Write-Host "Setting up Python PATH..." -ForegroundColor Green
Write-Host ""

# Check if Anaconda exists
if (-not (Test-Path $anacondaPath)) {
    Write-Host "Error: Anaconda not found at $anacondaPath" -ForegroundColor Red
    exit 1
}

# Add to current session PATH
$env:Path = "$anacondaPath;$anacondaScripts;$env:Path"

Write-Host "Added Anaconda to PATH for current session" -ForegroundColor Green
Write-Host ""

# Try to add to user PATH permanently (requires no admin for user PATH)
try {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($userPath -notlike "*$anacondaPath*") {
        $newUserPath = "$anacondaPath;$anacondaScripts;$userPath"
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "Added Anaconda to User PATH permanently!" -ForegroundColor Green
    } else {
        Write-Host "Anaconda already in User PATH" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not update User PATH permanently. You may need to run as Administrator." -ForegroundColor Yellow
    Write-Host "For now, Anaconda is added to this session only." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing Python..." -ForegroundColor Cyan
python --version

Write-Host ""
Write-Host "Setup complete! You can now use 'python server.py' to run the server." -ForegroundColor Green
Write-Host "Note: If PATH changes don't work, restart your terminal." -ForegroundColor Yellow

