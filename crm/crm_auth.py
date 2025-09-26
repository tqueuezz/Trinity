"""
crm_auth.py - Authentication and business context switching scaffolding
This module provides pseudocode/starter implementations for:
- user registration and login
- password hashing/verification
- multi-business membership & role-based permissions
- business context switching within a session

Note: This is framework-agnostic scaffolding. Wire it to your DB/session layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
import hashlib
import hmac
import os
import secrets

# ---- Roles & Permissions ----
class Role(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MANAGER = "manager"
    STAFF = "staff"
    VIEWER = "viewer"

# Simple, extensible permission map (can be moved to DB)
DEFAULT_ROLE_PERMISSIONS: Dict[str, List[str]] = {
    Role.OWNER.value: ["*"],
    Role.ADMIN.value: [
        "contacts:read", "contacts:write",
        "leads:read", "leads:write",
        "pipeline:read", "pipeline:write",
        "settings:read", "settings:write",
        "billing:read", "billing:write",
    ],
    Role.MANAGER.value: [
        "contacts:read", "contacts:write",
        "leads:read", "leads:write",
        "pipeline:read", "pipeline:write",
        "settings:read",
    ],
    Role.STAFF.value: [
        "contacts:read", "contacts:write",
        "leads:read", "leads:write",
        "pipeline:read",
    ],
    Role.VIEWER.value: [
        "contacts:read", "leads:read", "pipeline:read"
    ],
}

# ---- Password hashing helpers (salted PBKDF2) ----
# For production, prefer libs like passlib/bcrypt/argon2.
PBKDF2_ITERATIONS = 210_000
HASH_NAME = "sha256"
SALT_BYTES = 16
KEY_LEN = 32


def _pbkdf2_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=KEY_LEN)


def hash_password(password: str) -> str:
    """Return 'pbkdf2$hexsalt$hexhash' string."""
    salt = os.urandom(SALT_BYTES)
    key = _pbkdf2_hash(password, salt)
    return "pbkdf2$%s$%s" % (salt.hex(), key.hex())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, hex_salt, hex_key = stored.split("$")
        if scheme != "pbkdf2":
            return False
        salt = bytes.fromhex(hex_salt)
        expected = bytes.fromhex(hex_key)
        computed = _pbkdf2_hash(password, salt)
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


# ---- Data structures (can be replaced by ORM models) ----
@dataclass
class BusinessMembership:
    business_id: str
    role: Role
    # Optional per-business overrides of permissions
    permissions_override: Optional[List[str]] = None


@dataclass
class User:
    user_id: str
    email: str
    password_hash: str
    # Users can belong to multiple businesses
    memberships: List[BusinessMembership] = field(default_factory=list)
    # Profile, settings, etc.
    profile: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, business_id: str, permission: str) -> bool:
        """Check permission under a specific business context.
        - '*' on role grants all permissions
        - permissions_override can grant/restrict
        """
        membership = next((m for m in self.memberships if m.business_id == business_id), None)
        if membership is None:
            return False
        if membership.role == Role.OWNER:
            return True
        # base perms from role map
        base = DEFAULT_ROLE_PERMISSIONS.get(membership.role.value, [])
        if "*" in base:
            return True
        # apply override if present
        if membership.permissions_override is not None:
            # If override contains explicit deny semantics in future, handle here
            base = membership.permissions_override
        return permission in base


# ---- In-memory stores (replace with DB) ----
USERS_BY_EMAIL: Dict[str, User] = {}
USERS_BY_ID: Dict[str, User] = {}


# ---- Registration / Login ----

def register_user(email: str, password: str) -> User:
    """Create a new user; idempotent by email for demo purposes."""
    if email in USERS_BY_EMAIL:
        return USERS_BY_EMAIL[email]
    user_id = secrets.token_hex(8)
    user = User(
        user_id=user_id,
        email=email.lower().strip(),
        password_hash=hash_password(password),
    )
    USERS_BY_EMAIL[user.email] = user
    USERS_BY_ID[user.user_id] = user
    return user


def login_user(email: str, password: str) -> Optional[User]:
    user = USERS_BY_EMAIL.get(email.lower().strip())
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# ---- Business membership management ----

def add_user_to_business(user: User, business_id: str, role: Role = Role.VIEWER,
                         permissions_override: Optional[List[str]] = None) -> None:
    if next((m for m in user.memberships if m.business_id == business_id), None):
        # already a member; update role/override
        for m in user.memberships:
            if m.business_id == business_id:
                m.role = role
                m.permissions_override = permissions_override
                return
    user.memberships.append(BusinessMembership(business_id=business_id, role=role, permissions_override=permissions_override))


# ---- Session/Context switching (pseudocode) ----
class Session(dict):
    """Very light session dict. Replace with Streamlit/Flask/Django session."""
    pass


def switch_business_context(session: Session, user: User, target_business_id: str) -> bool:
    """Switch the active business context if the user is a member of the target business."""
    if next((m for m in user.memberships if m.business_id == target_business_id), None) is None:
        return False
    session["active_business_id"] = target_business_id
    return True


def current_business_context(session: Session) -> Optional[str]:
    return session.get("active_business_id")


# ---- Example integration notes (to wire with UI/layout) ----
"""
How to integrate with Streamlit UI:

import streamlit as st
from crm.crm_auth import (login_user, register_user, switch_business_context, current_business_context, Role)

# During login
user = login_user(email, password)
if user:
    st.session_state["user_id"] = user.user_id
    # choose default business context, e.g. first membership
    if user.memberships:
        st.session_state["active_business_id"] = user.memberships[0].business_id

# Business switcher UI
biz_id = st.selectbox("Business", [m.business_id for m in user.memberships])
if st.button("Switch"):
    switch_business_context(st.session_state, user, biz_id)

# Permissions-based UI visibility example
can_manage_leads = user.has_permission(current_business_context(st.session_state), "leads:write")
if can_manage_leads:
    st.button("Create Lead")
"""
