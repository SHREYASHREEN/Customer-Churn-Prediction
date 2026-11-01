"""
auth.py - user registration and login for the Streamlit app.

Security design (all from Python's standard library, nothing extra to install):
* Passwords are NEVER stored. Only a salted hash is stored: PBKDF2-HMAC-SHA256 with a random
  16-byte salt per user and 600,000 iterations.
* Hashes are compared with hmac.compare_digest (constant-time).
* All database access uses parameterised SQL (no SQL injection).
* After 5 wrong passwords the account is locked for 5 minutes (stored in the database).
* The login error message is the same for "unknown user" and "wrong password".

Users live in a small SQLite file: auth_data/users.db (created automatically on first use).
Set the environment variable CHURN_AUTH_DB to use a different file (the tests do this).

Limitations (fine for a college demo, not for production): the login lasts only while the browser tab
session lasts (a page refresh logs you out), there is no password-reset or e-mail verification.
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
MAX_FAILED_ATTEMPTS = 5
LOCK_SECONDS = 300
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
SESSION_KEY = "auth_user"


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------
def hash_password(password, salt=None, iterations=PBKDF2_ITERATIONS):
    """Return 'pbkdf2_sha256$<iterations>$<salt>$<hash>' (all base64). The password itself is not kept."""
    salt = salt if salt is not None else secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join(["pbkdf2_sha256", str(iterations),
                     base64.b64encode(salt).decode("ascii"), base64.b64encode(digest).decode("ascii")])


def verify_password(password, stored):
    """True if `password` matches the stored hash string."""
    try:
        scheme, iterations, salt_b64, hash_b64 = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(hash_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# a fixed hash used to spend the same time when the username does not exist (hides which usernames exist)
_DUMMY_HASH = hash_password("not-a-real-password")


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
def db_path():
    override = os.environ.get("CHURN_AUTH_DB")
    return Path(override) if override else Path(__file__).resolve().parent / "auth_data" / "users.db"


def _connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               username TEXT NOT NULL UNIQUE COLLATE NOCASE,
               full_name TEXT NOT NULL,
               password_hash TEXT NOT NULL,
               created_at TEXT NOT NULL,
               failed_attempts INTEGER NOT NULL DEFAULT 0,
               locked_until REAL NOT NULL DEFAULT 0)""")
    return conn


# ---------------------------------------------------------------------------
# registration / login logic (no Streamlit inside, so it can be tested directly)
# ---------------------------------------------------------------------------
def validate_registration(username, full_name, password, confirm):
    """Return a list of problems with the registration form (empty list = OK)."""
    problems = []
    if not full_name.strip():
        problems.append("Please enter your full name.")
    elif len(full_name.strip()) > 80:
        problems.append("Full name is too long (maximum 80 characters).")
    if not USERNAME_PATTERN.match(username):
        problems.append("Username must be 3-30 characters: letters, digits, '_', '.' or '-' only.")
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    elif not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        problems.append("Password must contain at least one letter and one digit.")
    if password != confirm:
        problems.append("The two passwords do not match.")
    return problems


def register_user(username, full_name, password, confirm):
    """Create an account. Returns (ok, message)."""
    username, full_name = username.strip(), full_name.strip()
    problems = validate_registration(username, full_name, password, confirm)
    if problems:
        return False, " ".join(problems)
    stored = hash_password(password)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, full_name, stored, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    except sqlite3.IntegrityError:
        return False, "That username is already taken. Please choose another one."
    return True, "Account created. You can now log in."


def authenticate(username, password):
    """Check a login. Returns (ok, message, user_dict_or_None)."""
    generic = "Invalid username or password."
    username = username.strip()
    if not username or not password:
        return False, generic, None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, username, full_name, password_hash, failed_attempts, locked_until FROM users WHERE username = ?",
            (username,)).fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)           # same work as a real check
            return False, generic, None
        user_id, name, full_name, stored, failed, locked_until = row
        now = time.time()
        if locked_until > now:
            minutes = int((locked_until - now) // 60) + 1
            return False, f"Too many failed attempts. This account is locked for about {minutes} more minute(s).", None
        if verify_password(password, stored):
            with conn:
                conn.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE id = ?", (user_id,))
            return True, "Login successful.", {"id": user_id, "username": name, "full_name": full_name}
        failed += 1
        with conn:
            if failed >= MAX_FAILED_ATTEMPTS:
                conn.execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?",
                             (now + LOCK_SECONDS, user_id))
            else:
                conn.execute("UPDATE users SET failed_attempts = ? WHERE id = ?", (failed, user_id))
        return False, generic, None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# session helpers
# ---------------------------------------------------------------------------
def is_logged_in():
    return st.session_state.get(SESSION_KEY) is not None


def current_user():
    return st.session_state.get(SESSION_KEY)


def logout():
    """Forget the user and every piece of page state, then show the login page."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# Streamlit screen
# ---------------------------------------------------------------------------
def render_auth_page():
    """Login / Register screen. The rest of the app is NOT shown until a user is logged in."""
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">📊 Customer Churn Prediction &amp; Retention Analysis</div>
          <div class="hero-sub">Machine-learning system for identifying customers at risk of leaving.
          Please log in or create an account to continue.</div>
        </div>
        """, unsafe_allow_html=True)

    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
            if submitted:
                ok, message, user = authenticate(username, password)
                if ok:
                    st.session_state[SESSION_KEY] = user
                    st.rerun()
                else:
                    st.error(message)

        with tab_register:
            with st.form("register_form"):
                full_name = st.text_input("Full name", key="reg_full_name")
                new_username = st.text_input("Choose a username", key="reg_username",
                                             help="3-30 characters: letters, digits, _ . -")
                new_password = st.text_input("Choose a password", type="password", key="reg_password",
                                             help="At least 8 characters, with a letter and a digit.")
                confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                created = st.form_submit_button("Create account", use_container_width=True)
            if created:
                ok, message = register_user(new_username, full_name, new_password, confirm)
                (st.success if ok else st.error)(message)
                if ok:
                    st.info("Open the **Login** tab and sign in with your new account.")
        st.caption("Passwords are stored only as salted PBKDF2-SHA256 hashes, never as plain text.")
