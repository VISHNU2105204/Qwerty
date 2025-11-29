# Email Setup Guide for OTP Functionality

## Quick Setup for Gmail

### Step 1: Enable 2-Step Verification
1. Go to your Google Account: https://myaccount.google.com/
2. Navigate to **Security** section
3. Enable **2-Step Verification** (if not already enabled)

### Step 2: Generate App Password
1. Go to: https://myaccount.google.com/apppasswords
2. Select **Mail** as the app
3. Select **Other (Custom name)** as device, enter "Truth Detector"
4. Click **Generate**
5. **Copy the 16-character password** (it will look like: `abcd efgh ijkl mnop`)
6. **Save it somewhere safe** - you won't see it again!

### Step 3: Configure Server

#### Option A: Environment Variables (Recommended)

**Windows PowerShell:**
```powershell
$env:SMTP_USERNAME='your.email@gmail.com'
$env:SMTP_PASSWORD='abcdefghijklmnop'  # Use the 16-char app password (no spaces)
```

**Windows CMD:**
```cmd
set SMTP_USERNAME=your.email@gmail.com
set SMTP_PASSWORD=abcdefghijklmnop
```

**Linux/Mac:**
```bash
export SMTP_USERNAME='your.email@gmail.com'
export SMTP_PASSWORD='abcdefghijklmnop'
```

Then start the server in the same terminal window.

#### Option B: Direct Configuration in server.py

Edit `server.py` and find the `send_email()` function. Uncomment and modify these lines:

```python
SMTP_USERNAME = 'your.email@gmail.com'  # Your Gmail address
SMTP_PASSWORD = 'abcdefghijklmnop'  # Your 16-character App Password (no spaces)
```

**Important:** Remove the `os.getenv()` lines or set them to use the direct values.

### Step 4: Test

1. Restart the server
2. Go to the forgot password page
3. Enter your email
4. Check your email inbox (and spam folder) for the OTP

---

## Other Email Providers

### Outlook/Hotmail
```python
SMTP_SERVER = 'smtp-mail.outlook.com'
SMTP_PORT = 587
# Use your email and password (may need App Password if 2FA enabled)
```

### Yahoo Mail
```python
SMTP_SERVER = 'smtp.mail.yahoo.com'
SMTP_PORT = 587
# Use your email and App Password
```

### Custom SMTP Server
```python
SMTP_SERVER = 'your.smtp.server.com'
SMTP_PORT = 587  # or 465 for SSL
SMTP_USERNAME = 'your.email@domain.com'
SMTP_PASSWORD = 'your-password'
```

---

## Troubleshooting

### Error: "SMTP Authentication Error"
- Make sure you're using an **App Password**, not your regular Gmail password
- Ensure 2-Step Verification is enabled
- Check that the password has no spaces (remove spaces from the 16-char code)

### Error: "Connection refused" or "Could not connect"
- Check your internet connection
- Verify firewall isn't blocking port 587
- Try different SMTP server/port combination

### Emails not received
- Check spam/junk folder
- Verify email address is correct
- Check server console for error messages
- For Gmail, check if "Less secure app access" is enabled (older accounts)

### Development/Testing
If you can't configure email, the OTP will be displayed:
- In the server console
- In the browser response (check browser console or network tab)
- On the forgot password page (for development only)

---

## Security Note

⚠️ **Never commit SMTP credentials to Git!**
- Use environment variables in production
- Don't hardcode passwords in server.py if you're sharing the code
- Consider using a secrets management service for production

