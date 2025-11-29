# How to Run the Server

## Quick Start

### Option 1: Use the Batch File (Easiest)
Double-click `run_server.bat` or run it from terminal:
```bash
run_server.bat
```

### Option 2: Use PowerShell Script
```powershell
.\run_server.ps1
```

### Option 3: Use Python Directly (After Setup)
1. First, run the setup script to add Python to PATH:
   ```powershell
   .\setup_python_path.ps1
   ```
   Or restart your terminal after running it.

2. Then run:
   ```bash
   python server.py
   ```

### Option 4: Use Full Path (Always Works)
```bash
"C:\Users\THARUN\anaconda3\python.exe" server.py
```

## Fixing Python Command Issues

If `python` command doesn't work, it's because Python is not in your PATH. 

### Temporary Fix (Current Session Only)
Run this in PowerShell:
```powershell
$env:Path = "C:\Users\THARUN\anaconda3;C:\Users\THARUN\anaconda3\Scripts;$env:Path"
```

### Permanent Fix
1. Run the setup script:
   ```powershell
   .\setup_python_path.ps1
   ```

2. Or manually add to PATH:
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables", select "Path" and click "Edit"
   - Add: `C:\Users\THARUN\anaconda3`
   - Add: `C:\Users\THARUN\anaconda3\Scripts`
   - Click OK on all dialogs
   - Restart your terminal

## Server Access

Once the server is running, access it at:
- **Main URL**: http://127.0.0.1:5500
- **Pages**:
  - http://127.0.0.1:5500/index.html
  - http://127.0.0.1:5500/simple_detect.html
  - http://127.0.0.1:5500/login.html
  - http://127.0.0.1:5500/profile.html

## Stopping the Server

Press `Ctrl + C` in the terminal where the server is running.

