"""
src/db/auth.py
--------------
User authentication, registration, and credential management routines.
"""

import hashlib
import sqlite3
from pathlib import Path

# Database setup
DB_PATH = Path(__file__).resolve().parent.parent.parent / "plagiarism_detector.db"


def init_db():
    """Initialize database tables and seed default admin user if empty."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
        """
    )
    conn.commit()

    # Seed default admin account if users table is completely empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        default_pass_hash = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", default_pass_hash, "admin"),
        )
        conn.commit()

    conn.close()


def add_user(username: str, password: str, role: str = "user") -> bool:
    """Add a new user to the database."""
    init_db()  # Ensure DB is initialized
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        raise ValueError(f"User '{username}' already exists.")
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Verify user credentials against stored password hashes."""
    init_db()  # Ensure DB is initialized
    if not username or not password:
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "SELECT username FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash),
    )
    user = cursor.fetchone()
    conn.close()

    return user is not None


# Alias for compatibility
authenticate_user = verify_user


def get_user_role(username: str) -> str:
    """Get the role of a given user ('admin' or 'user')."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else "user"


def get_all_users() -> list:
    """Retrieve all registered users and their roles."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    users = cursor.fetchall()
    conn.close()

    return [{"username": u[0], "role": u[1]} for u in users]


def delete_user(username: str) -> bool:
    """Delete a user from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return deleted


def update_password(username: str, new_password: str) -> bool:
    """Update password for an existing user."""
    init_db()
    if not new_password or len(new_password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (password_hash, username),
    )
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()

    if updated_rows == 0:
        raise ValueError("User not found.")

    return True