"""
User model for multi-user authentication.
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path


class User(UserMixin):
    """
    User model for authentication and authorization.
    """
    
    def __init__(self, user_id: str, username: str, email: str, password_hash: str, 
                 role: str = "user", is_active: bool = True, created_at: str = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self._is_active = is_active
        self.created_at = created_at
    
    def get_id(self):
        """Return the user ID for Flask-Login."""
        return self.user_id
    
    @property
    def is_active(self):
        """Return whether the user is active (required by Flask-Login)."""
        return self._is_active
    
    def check_password(self, password: str) -> bool:
        """Check if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)
    
    def set_password(self, password: str):
        """Set a new password hash."""
        self.password_hash = generate_password_hash(password)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for storage."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'role': self.role,
            'is_active': self._is_active,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            role=data.get('role', 'user'),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at')
        )
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        if self.role == 'admin':
            return True
        elif self.role == 'trader':
            return permission in ['view_dashboard', 'view_trades', 'modify_parameters']
        elif self.role == 'viewer':
            return permission in ['view_dashboard', 'view_trades']
        return False


class UserManager:
    """
    Manages user storage and authentication.
    """
    
    def __init__(self, users_file: str = "users.json"):
        self.users_file = Path(users_file)
        self.users = {}
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.values():
                        user = User.from_dict(user_data)
                        self.users[user.user_id] = user
            except Exception as e:
                print(f"Error loading users: {e}")
                self.create_default_users()
        else:
            self.create_default_users()
    
    def save_users(self):
        """Save users to JSON file."""
        try:
            data = {user_id: user.to_dict() for user_id, user in self.users.items()}
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def create_default_users(self):
        """Create default users if no users file exists."""
        from datetime import datetime
        
        # Create admin user
        admin = User(
            user_id="admin",
            username="admin",
            email="admin@trading.com",
            password_hash=generate_password_hash("admin123"),
            role="admin",
            created_at=datetime.now().isoformat()
        )
        
        # Create trader user
        trader = User(
            user_id="trader",
            username="trader",
            email="trader@trading.com",
            password_hash=generate_password_hash("trader123"),
            role="trader",
            created_at=datetime.now().isoformat()
        )
        
        # Create viewer user
        viewer = User(
            user_id="viewer",
            username="viewer",
            email="viewer@trading.com",
            password_hash=generate_password_hash("viewer123"),
            role="viewer",
            created_at=datetime.now().isoformat()
        )
        
        self.users = {
            "admin": admin,
            "trader": trader,
            "viewer": viewer
        }
        self.save_users()
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def add_user(self, user: User):
        """Add a new user."""
        self.users[user.user_id] = user
        self.save_users()
    
    def update_user(self, user: User):
        """Update an existing user."""
        if user.user_id in self.users:
            self.users[user.user_id] = user
            self.save_users()
    
    def delete_user(self, user_id: str):
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
            self.save_users()
    
    def list_users(self) -> Dict[str, User]:
        """Get all users."""
        return self.users.copy()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = self.get_user_by_username(username)
        if user and user._is_active and user.check_password(password):
            return user
        return None
