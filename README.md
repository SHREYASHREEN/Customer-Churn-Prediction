# ChurnIQ — Customer Churn Prediction

A Streamlit-based customer churn prediction project using a trained XGBoost model.

## What's new in this updated version

- Login and logout
- Create-account / registration page
- Local SQLite user database (`users.db` is created automatically on first run)
- Passwords stored as salted PBKDF2-HMAC-SHA256 hashes (not plain text)
- Session-based access to the prediction dashboard
- Refreshed dashboard layout and customer-profile sections
- Churn probability, Low/Medium/High risk band, and prediction result

## Run on Windows (VS Code / PowerShell)

1. Extract this ZIP file.
2. Open the extracted `Customer_Churn_Project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this in the same terminal and retry activation:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. Streamlit will show a local address, usually `http://localhost:8501`, and may open it in your browser.
8. Select **Create account**, make a username and password (at least 8 characters), then go to **Login**.
9. After login, enter customer details and click **Predict churn**. Use **Logout** in the sidebar to end the session.

## Important notes

- Keep the trained `.pkl` model files in the project root as supplied.
- `users.db` is generated automatically when you first run the app. It stores local demo accounts. It is not included in the ZIP.
- If you share the project folder with a mentor, they can register their own account on their machine.
- This authentication is suitable for a local academic demonstration, not production deployment. A production application should use HTTPS, managed identity/authentication, rate limiting, secure secret management, and additional account protections.
- The model prediction is an estimate based on the training data and should not be treated as a guarantee.
