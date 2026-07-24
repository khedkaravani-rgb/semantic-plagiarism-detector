import json
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
import json
import hashlib
import secrets

from src.db.migrations import migrate_auth_database

_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "users.db")
)


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, check_same_thread=False)
def _hash_password(password: str) -> str:
    """Return a bcrypt hash for the given password."""
    return bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt(10),
    ).decode()


def _hash_password(password: str, salt: str = None) -> str:
    """Return a sha256 hash for the given password."""
    if salt is None:
        salt = secrets.token_hex(8)
    # Simple hash for local dev to avoid bcrypt DLL hell
    pwd_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def _validate_username(username: str) -> str:
    username = str(username).strip().lower()
    if not username:
        raise ValueError("Username cannot be empty.")
    return username


def _validate_password(password: str) -> str:
    try:
        password = str(password)
        if len(password.strip()) < 5:
            raise ValueError("Password must be at least 5 characters long.")
        return password
    finally:
        password = "REDACTED"


def _validate_role(role: str) -> str:
    role = str(role).strip().lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return role


def init_db() -> None:
    """Create users table and seed default admin if not exists."""
    with _connect() as conn:
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
        
        # Schema migration: add tour_completed column if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tour_completed" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN tour_completed INTEGER DEFAULT 0")
            conn.commit()
        
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if not exists:
            hashed = _hash_password("admin123")
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", hashed, "admin"),
            )
        conn.commit()


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists and password matches the stored hash."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    if not row:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())
    try:
        username = _validate_username(username)
        password = _validate_password(password)
    except ValueError:
        return False

    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return False

    stored_hash = row[0]
    
    # Handle old bcrypt hashes (mock verification for local dev) or new sha256 hashes
    if stored_hash.startswith("$2") or stored_hash.startswith("$1"):
        # We can't verify old bcrypt hashes without the bcrypt library.
        # So we just accept the default admin password for convenience.
        if username == "admin" and password == "admin123":
            return True
        return False
        
    try:
        salt, _ = stored_hash.split("$", 1)
        return stored_hash == _hash_password(password, salt)
    except ValueError:
        return False


# Alias for compatibility
authenticate_user = verify_user



def get_user_role(username: str) -> str | None:
    """Return the role of a user, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return row[0] if row else None


def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a new user with a bcrypt-hashed password."""
    hashed = _hash_password(password)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username.lower(), hashed, role),
        )
        conn.commit()

    return row[0] if row else None


def get_or_create_sso_user(email: str) -> str:
    """Return the role of an SSO user, creating them as a teacher if they don't exist."""
    role = get_user_role(email)
    if role:
        return role
        
    # Generate a random password since they authenticate via SSO
    import secrets
    random_password = secrets.token_urlsafe(32)
    add_user(email, random_password, "teacher")
    return "teacher"



def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a user and preserve SQLite duplicate-user semantics."""
    try:
        username = _validate_username(username)
        password = _validate_password(password)
        role = _validate_role(role)

        hashed = _hash_password(password)

        with _connect() as conn:
            # The UNIQUE constraint is the source of truth. Existing callers and
            # tests rely on sqlite3.IntegrityError for duplicate usernames.
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role),
            )
            conn.commit()
    finally:
        password = "REDACTED"


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
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role FROM users ORDER BY id"
        ).fetchall()
    return [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]


def delete_user(username: str) -> None:
    """Delete a user by username."""
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username.lower(),))
        conn.commit()


def update_password(username: str, new_password: str) -> None:
    """Update a user's password with a new bcrypt hash."""
    hashed = _hash_password(new_password)
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hashed, username.lower()),
        )
        conn.commit()


def get_tour_completed(username: str) -> bool:
    """Return whether a user has completed the onboarding tour."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT tour_completed FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return bool(row[0]) if row else False


def set_tour_completed(username: str, completed: bool = True) -> None:
    """Mark a user as having completed the onboarding tour."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET tour_completed = ? WHERE username = ?",
            (1 if completed else 0, username.lower()),
        )
        conn.commit()

def get_2fa_status(username: str) -> tuple[bool, str | None]:
    """Return (two_factor_enabled, otp_secret) for a user."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT two_factor_enabled, otp_secret FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()
    if not row:
        return False, None
    return bool(row[0]), row[1]


def enable_2fa(username: str, secret: str) -> None:
    """Enable 2FA for a user and store their OTP secret."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 1, otp_secret = ? WHERE username = ?",
            (secret, username.lower()),
        )
        conn.commit()


def disable_2fa(username: str) -> None:
    """Disable 2FA for a user and clear their OTP secret."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 0, otp_secret = NULL WHERE username = ?",
            (username.lower(),),
        )
        conn.commit()


def check_login_rate_limit(username: str) -> tuple[bool, str | None]:
    """Check if username is rate limited. Returns (is_allowed, error_message)."""
    from src.utils.redis_cache import is_login_locked_out, get_login_attempts
    
    identifier = username.lower()
    if is_login_locked_out(identifier):
        attempts = get_login_attempts(identifier)
        return False, f"Account locked due to too many failed attempts. Please try again in 15 minutes. ({attempts}/5 attempts)"
    return True, None


def record_failed_login(username: str) -> None:
    """Record a failed login attempt for rate limiting."""
    from src.utils.redis_cache import increment_login_attempts
    identifier = username.lower()
    increment_login_attempts(identifier)


def clear_login_attempts(username: str) -> None:
    """Clear failed login attempts after successful login."""
    from src.utils.redis_cache import clear_login_attempts as redis_clear_login_attempts
    identifier = username.lower()
    redis_clear_login_attempts(identifier)


def get_user_preferences(username: str) -> dict:
    """Return user preferences as a dictionary, or empty dict if none exist."""
    username = username.lower()

    with _connect() as conn:
        row = conn.execute(
            "SELECT preferences FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row and row[0]:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}
    return {}


def update_user_preferences(username: str, preferences: dict) -> None:
    """Serialize and update user preferences in the database."""
    username = username.lower()
    prefs_str = json.dumps(preferences)

    with _connect() as conn:
        conn.execute(
            "UPDATE users SET preferences = ? WHERE username = ?",
            (prefs_str, username),
        )
        conn.commit()


def get_or_create_sso_user(email: str, default_role: str = "teacher") -> str:
    """Finds a user by email (as username) or creates a new one for SSO."""
    username = _validate_username(email)
    
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        
        if row:
            return row[0]
            
        # Create user with dummy password
        hashed = _hash_password("!")
        role = _validate_role(default_role)
        
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed, role),
        )
        conn.commit()
        return role
