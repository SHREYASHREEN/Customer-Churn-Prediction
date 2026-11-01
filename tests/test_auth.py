"""
test_auth.py - tests of registration, login, logout and page protection.

Run from the project folder:   python tests/test_auth.py
(Uses a temporary user database; your real auth_data/users.db is never touched.)
"""

import os
import sqlite3
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
TMP_DB = Path(tempfile.mkdtemp()) / "test_users.db"
os.environ["CHURN_AUTH_DB"] = str(TMP_DB)

import auth  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

results = []


def record(ok, name, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


# =============================================================================== A. hashing
print("=== A. Password hashing ===")
h1, h2 = auth.hash_password("Secret123"), auth.hash_password("Secret123")
record(h1 != h2, "The same password gives different hashes (random salt per user)")
record(h1.startswith("pbkdf2_sha256$600000$"), "Hash uses PBKDF2-SHA256 with 600,000 iterations")
record("Secret123" not in h1, "The hash does not contain the plain password")
record(auth.verify_password("Secret123", h1) and not auth.verify_password("secret123", h1) and not auth.verify_password("", h1),
       "Correct password verifies; wrong-case and empty passwords do not")
record(not auth.verify_password("Secret123", "garbage") and not auth.verify_password("Secret123", "a$b$c$d")
       and not auth.verify_password("Secret123", h1[:-4] + "AAAA"), "Malformed or altered hash strings never verify (and never crash)")

# =============================================================================== B. registration rules
print("\n=== B. Registration rules ===")
for label, args, needle in [
    ("username too short", ("ab", "Name", "Passw0rd1", "Passw0rd1"), "Username must be"),
    ("username with spaces/symbols", ("bad name!", "Name", "Passw0rd1", "Passw0rd1"), "Username must be"),
    ("password too short", ("gooduser", "Name", "Pw1", "Pw1"), "at least 8"),
    ("password without a digit", ("gooduser", "Name", "PasswordOnly", "PasswordOnly"), "letter and one digit"),
    ("password without a letter", ("gooduser", "Name", "12345678", "12345678"), "letter and one digit"),
    ("passwords differ", ("gooduser", "Name", "Passw0rd1", "Passw0rd2"), "do not match"),
    ("empty full name", ("gooduser", "  ", "Passw0rd1", "Passw0rd1"), "full name"),
]:
    ok, message = auth.register_user(*args)
    record(not ok and needle.lower() in message.lower(), f"Rejected: {label}")
ok, message = auth.register_user("alice_1", "Alice Example", "Passw0rd1", "Passw0rd1")
record(ok, "Valid registration is accepted")
ok, message = auth.register_user("ALICE_1", "Other Person", "Passw0rd9", "Passw0rd9")
record(not ok and "already taken" in message, "Duplicate username is rejected (case-insensitive)")

raw = TMP_DB.read_bytes()
record(b"Passw0rd1" not in raw, "The database file does not contain the plain-text password anywhere")
with sqlite3.connect(TMP_DB) as conn:
    stored = conn.execute("SELECT password_hash FROM users WHERE username='alice_1'").fetchone()[0]
record(stored.startswith("pbkdf2_sha256$") and "Passw0rd1" not in stored, "Stored value is a salted PBKDF2 hash")

# =============================================================================== C. login logic
print("\n=== C. Login, SQL injection and lockout ===")
ok, message, user = auth.authenticate("alice_1", "Passw0rd1")
record(ok and user["full_name"] == "Alice Example", "Correct login succeeds")
ok, message, _ = auth.authenticate("ALICE_1", "Passw0rd1")
record(ok, "Username is not case-sensitive")
ok_wrong, msg_wrong, _ = auth.authenticate("alice_1", "wrongpass1")
ok_unknown, msg_unknown, _ = auth.authenticate("nobody", "wrongpass1")
record(not ok_wrong and not ok_unknown and msg_wrong == msg_unknown, "Wrong password and unknown user give the SAME message (no user enumeration)", msg_wrong)
for attack in ["' OR '1'='1", "alice_1'--", "alice_1' OR '1'='1' --", "x'; DROP TABLE users; --"]:
    ok, _, _ = auth.authenticate(attack, "anything1")
    ok2, _, _ = auth.authenticate("alice_1", attack)
    record(not ok and not ok2, f"SQL-injection attempt does not log in: {attack!r}")
# the attempts above also counted as failed logins for alice_1, so look at the table directly (not through login)
with sqlite3.connect(TMP_DB) as conn:
    n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    alice_row = conn.execute("SELECT username FROM users WHERE username='alice_1'").fetchone()
record(n_users == 1 and alice_row is not None, "The users table is intact after the injection attempts (nothing dropped, nothing added)")


def account_state():
    with sqlite3.connect(TMP_DB) as conn:
        return conn.execute("SELECT failed_attempts, locked_until FROM users WHERE username='alice_1'").fetchone()


def reset_account():
    with sqlite3.connect(TMP_DB) as conn:
        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE username='alice_1'")


# step-by-step lockout, starting from a clean account
reset_account()
for _ in range(auth.MAX_FAILED_ATTEMPTS - 1):
    auth.authenticate("alice_1", "wrongpass1")
failed, locked_until = account_state()
record(failed == auth.MAX_FAILED_ATTEMPTS - 1 and locked_until == 0, f"After {auth.MAX_FAILED_ATTEMPTS - 1} wrong passwords the account is NOT locked yet", f"failed_attempts={failed}")
auth.authenticate("alice_1", "wrongpass1")
failed, locked_until = account_state()
record(locked_until > time.time(), f"The {auth.MAX_FAILED_ATTEMPTS}th wrong password locks the account", f"locked for {int(locked_until - time.time())} s")
ok, message, _ = auth.authenticate("alice_1", "Passw0rd1")
record(not ok and "locked" in message, "While locked, even the CORRECT password is refused", message[:70])
with sqlite3.connect(TMP_DB) as conn:
    conn.execute("UPDATE users SET locked_until = ? WHERE username='alice_1'", (time.time() - 1,))
record(auth.authenticate("alice_1", "Passw0rd1")[0] and account_state() == (0, 0), "After the lock time has passed the user can log in again and the counter is reset")
reset_account()
auth.authenticate("alice_1", "wrongpass1")
auth.authenticate("alice_1", "Passw0rd1")
record(account_state()[0] == 0, "A successful login resets the failed-attempt counter")

# =============================================================================== D. the real app
print("\n=== D. Login flow and page protection in the real app ===")


def fresh_app():
    return AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120).run()


def set_login(at, username, password):
    at.text_input(key="login_username").set_value(username)
    at.text_input(key="login_password").set_value(password)
    at.button[0].click().run()


at = fresh_app()
record(not at.exception and [t.label for t in at.tabs] == ["🔐 Login", "📝 Register"] and len(at.sidebar.radio) == 0 and len(at.metric) == 0,
       "Logged-out visitors see only the Login/Register screen (no navigation, no dashboard)")

# direct access attempt: pretend the visitor already chose the prediction page
at = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120)
at.session_state["nav"] = "🔮 Predict Customer"
at.run()
record(not at.exception and len(at.selectbox) == 0 and len(at.sidebar.radio) == 0 and "Login" in " ".join(t.label for t in at.tabs),
       "Trying to open a page without logging in still shows only the login screen")

# registration through the UI
at = fresh_app()
for key, value in (("reg_full_name", "Ravi Kumar"), ("reg_username", "ravi_k"), ("reg_password", "Passw0rd77"), ("reg_confirm", "Passw0rd77")):
    at.text_input(key=key).set_value(value)
at.button[1].click().run()
record(any("Account created" in s.value for s in at.success), "Registration through the form works")
at = fresh_app()
for key, value in (("reg_full_name", "Ravi Again"), ("reg_username", "ravi_k"), ("reg_password", "Passw0rd77"), ("reg_confirm", "Passw0rd77")):
    at.text_input(key=key).set_value(value)
at.button[1].click().run()
record(any("already taken" in e.value for e in at.error), "Registering the same username again shows an error")
at = fresh_app()
for key, value in (("reg_full_name", "Weak Pw"), ("reg_username", "weak_pw"), ("reg_password", "short"), ("reg_confirm", "short")):
    at.text_input(key=key).set_value(value)
at.button[1].click().run()
record(any("at least 8" in e.value for e in at.error), "A weak password is refused with an explanation")

# login failure and success
at = fresh_app()
set_login(at, "ravi_k", "WrongPass1")
record(any("Invalid username or password" in e.value for e in at.error) and len(at.sidebar.radio) == 0,
       "Wrong password: error message and the app stays locked")
at = fresh_app()
set_login(at, "ravi_k", "Passw0rd77")
record(not at.exception and len(at.sidebar.radio) == 1 and len(at.metric) == 5, "Correct login opens the dashboard (5 KPI cards)")
sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
record("Ravi Kumar" in sidebar_text and "ravi_k" in sidebar_text, "The sidebar shows the logged-in user's name")

# every page is reachable when logged in
pages_ok = True
for page in at.sidebar.radio(key="nav").options:
    at.sidebar.radio(key="nav").set_value(page).run()
    pages_ok &= not at.exception
record(pages_ok, "All 7 pages open without errors for a logged-in user")

# logout
at.sidebar.radio(key="nav").set_value("🔮 Predict Customer").run()
at.sidebar.button(key="logout_button").click().run()
record(not at.exception and len(at.sidebar.radio) == 0 and len(at.selectbox) == 0 and [t.label for t in at.tabs] == ["🔐 Login", "📝 Register"],
       "Logout returns to the login screen and hides every protected page")
record("auth_user" not in at.session_state and "nav" not in at.session_state, "After logout the login session and page state are erased")

print(f"\nSUMMARY: {sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
