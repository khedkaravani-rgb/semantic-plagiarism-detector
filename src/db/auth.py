"""
auth.py
-------
SQLite-backed authentication with bcrypt password hashing.

Public API
----------
init_db()                          → create tables + seed default admin
verify_user(username, password)    → bool
get_user_role(username)            → str | None
add_user(username, password, role) → None
get_all_users()                    → list[dict]
delete_user(username)              → None
update_password(username, password)→ None
get_tour_completed(username)       → bool
set_tour_completed(username, completed) → None
"""

import sqlite3
import bcrypt
import os

_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "users.db")
)


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, check_same_thread=False)


def init_db() -> None:
    """Create users table and seed default admin if not exists."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'teacher',
                tour_completed INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]

        if "tour_completed" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN tour_completed INTEGER DEFAULT 0"
            )
            conn.commit()

        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", ("admin",)
        ).fetchone()

        if not exists:
            hashed = bcrypt.hashpw(b"admin123", bcrypt.gensalt(10)).decode()
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", hashed, "admin"),
            )

        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(
            f"Failed to initialize authentication database: {e}"
        ) from e
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists and password matches the stored hash."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()

        if not row:
            return False

        return bcrypt.checkpw(password.encode(), row[0].encode())

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to verify user: {e}") from e
    finally:
        conn.close()


def get_user_role(username: str) -> str | None:
    """Return the role of a user, or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()

        return row[0] if row else None

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve user role: {e}") from e
    finally:
        conn.close()


def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a new user with a bcrypt-hashed password."""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(10)).decode()

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username.lower(), hashed, role),
        )
        conn.commit()

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise ValueError(f"Username '{username}' already exists.") from e

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Failed to add user: {e}") from e

    finally:
        conn.close()


def get_all_users() -> list:
    """Return all users as a list of dicts (excludes password hashes)."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, username, role FROM users ORDER BY id"
        ).fetchall()

        return [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve users: {e}") from e

    finally:
        conn.close()


def delete_user(username: str) -> None:
    """Delete a user by username."""
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM users WHERE username = ?",
            (username.lower(),),
        )
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Failed to delete user: {e}") from e

    finally:
        conn.close()


def update_password(username: str, new_password: str) -> None:
    """Update a user's password with a new bcrypt hash."""
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt(10)).decode()

    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hashed, username.lower()),
        )
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Failed to update password: {e}") from e

    finally:
        conn.close()


def get_tour_completed(username: str) -> bool:
    """Return whether a user has completed the onboarding tour."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT tour_completed FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()

        return bool(row[0]) if row else False

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve tour status: {e}") from e

    finally:
        conn.close()


def set_tour_completed(username: str, completed: bool = True) -> None:
    """Mark a user as having completed the onboarding tour."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET tour_completed = ? WHERE username = ?",
            (1 if completed else 0, username.lower()),
        )
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Failed to update tour status: {e}") from e

    finally:
        conn.close()
