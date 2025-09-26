"""CRM Utility Functions for Trinity Strategy & Marketing

This module provides in-memory CRUD utilities for contacts, leads, and business
records. It is designed to be easily swappable with a persistent backend later
(e.g., SQLite, Supabase, MongoDB) while offering a clean API now.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from datetime import datetime
import uuid

from .crm_models import (
    Contact,
    Lead,
    Business,
    ContactStatus,
    LeadStatus,
    PipelineStage,
    BusinessType,
)


class InMemoryStore:
    """Simple in-memory store to scaffold data operations.

    Replace with a real repository later. Methods mirror common CRUD patterns.
    """

    def __init__(self):
        self.contacts: Dict[str, Contact] = {}
        self.leads: Dict[str, Lead] = {}
        self.businesses: Dict[str, Business] = {}

    # ---------------------- CONTACTS ----------------------
    def add_contact(self, contact: Contact) -> Contact:
        contact_id = contact.contact_id or str(uuid.uuid4())
        contact.contact_id = contact_id
        contact.created_at = contact.created_at or datetime.now()
        contact.updated_at = datetime.now()
        self.contacts[contact_id] = contact
        return contact

    def update_contact(self, contact_id: str, **updates) -> Optional[Contact]:
        c = self.contacts.get(contact_id)
        if not c:
            return None
        for k, v in updates.items():
            if hasattr(c, k):
                setattr(c, k, v)
        c.updated_at = datetime.now()
        return c

    def get_contact(self, contact_id: str) -> Optional[Contact]:
        return self.contacts.get(contact_id)

    def search_contacts(self, query: str = "", filters: Dict[str, Any] | None = None) -> List[Contact]:
        filters = filters or {}
        q = query.lower().strip()
        results: List[Contact] = []
        for c in self.contacts.values():
            blob = " ".join(
                [
                    c.email or "",
                    c.phone or "",
                    c.first_name or "",
                    c.last_name or "",
                    c.company or "",
                    c.source or "",
                    " ".join(c.tags or []),
                    c.notes or "",
                ]
            ).lower()
            if q and q not in blob:
                continue
            # filter matching
            ok = True
            for fk, fv in filters.items():
                val = getattr(c, fk, None)
                if isinstance(val, list):
                    ok = fv in val
                else:
                    ok = val == fv
                if not ok:
                    break
            if ok:
                results.append(c)
        return results

    # ---------------------- LEADS ----------------------
    def add_lead(self, lead: Lead) -> Lead:
        lead_id = lead.lead_id or str(uuid.uuid4())
        lead.lead_id = lead_id
        lead.created_at = lead.created_at or datetime.now()
        lead.updated_at = datetime.now()
        self.leads[lead_id] = lead
        return lead

    def update_lead(self, lead_id: str, **updates) -> Optional[Lead]:
        l = self.leads.get(lead_id)
        if not l:
            return None
        for k, v in updates.items():
            if hasattr(l, k):
                setattr(l, k, v)
        l.updated_at = datetime.now()
        return l

    def get_lead(self, lead_id: str) -> Optional[Lead]:
        return self.leads.get(lead_id)

    def search_leads(self, query: str = "", filters: Dict[str, Any] | None = None) -> List[Lead]:
        filters = filters or {}
        q = query.lower().strip()
        results: List[Lead] = []
        for l in self.leads.values():
            blob = " ".join(
                [
                    l.title or "",
                    l.description or "",
                    l.source or "",
                    l.campaign or "",
                    l.status.value if isinstance(l.status, LeadStatus) else str(l.status),
                    l.pipeline_stage.value if isinstance(l.pipeline_stage, PipelineStage) else str(l.pipeline_stage),
                ]
            ).lower()
            if q and q not in blob:
                continue
            ok = True
            for fk, fv in filters.items():
                val = getattr(l, fk, None)
                if isinstance(val, Enum):
                    val = val.value
                ok = val == fv
                if not ok:
                    break
            if ok:
                results.append(l)
        return results

    # ---------------------- BUSINESSES ----------------------
    def add_business(self, business: Business) -> Business:
        business_id = business.business_id or str(uuid.uuid4())
        business.business_id = business_id
        business.created_at = business.created_at or datetime.now()
        business.updated_at = datetime.now()
        self.businesses[business_id] = business
        return business

    def update_business(self, business_id: str, **updates) -> Optional[Business]:
        b = self.businesses.get(business_id)
        if not b:
            return None
        for k, v in updates.items():
            if hasattr(b, k):
                setattr(b, k, v)
        b.updated_at = datetime.now()
        return b

    def get_business(self, business_id: str) -> Optional[Business]:
        return self.businesses.get(business_id)

    def search_businesses(self, query: str = "", filters: Dict[str, Any] | None = None) -> List[Business]:
        filters = filters or {}
        q = query.lower().strip()
        results: List[Business] = []
        for b in self.businesses.values():
            blob = " ".join(
                [
                    b.name or "",
                    b.website or "",
                    b.industry or "",
                    " ".join(b.marketing_segments or []),
                ]
            ).lower()
            if q and q not in blob:
                continue
            ok = True
            for fk, fv in filters.items():
                val = getattr(b, fk, None)
                if isinstance(val, list):
                    ok = fv in val
                else:
                    ok = val == fv
                if not ok:
                    break
            if ok:
                results.append(b)
        return results


# Convenience factory
_store: Optional[InMemoryStore] = None

def get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


# High-level helper functions (thin wrappers)

def add_contact(**kwargs) -> Contact:
    c = Contact(**kwargs)
    return get_store().add_contact(c)

def update_contact(contact_id: str, **updates) -> Optional[Contact]:
    return get_store().update_contact(contact_id, **updates)

def find_contacts(query: str = "", **filters) -> List[Contact]:
    return get_store().search_contacts(query=query, filters=filters or None)


def add_lead(**kwargs) -> Lead:
    l = Lead(**kwargs)
    return get_store().add_lead(l)

def update_lead(lead_id: str, **updates) -> Optional[Lead]:
    return get_store().update_lead(lead_id, **updates)

def find_leads(query: str = "", **filters) -> List[Lead]:
    return get_store().search_leads(query=query, filters=filters or None)


def add_business(**kwargs) -> Business:
    b = Business(**kwargs)
    return get_store().add_business(b)

def update_business(business_id: str, **updates) -> Optional[Business]:
    return get_store().update_business(business_id, **updates)

def find_businesses(query: str = "", **filters) -> List[Business]:
    return get_store().search_businesses(query=query, filters=filters or None)


# Seed helpers for plug-and-play demos

def seed_demo_data() -> Dict[str, Any]:
    """Seed a minimal dataset suitable for marketing automation demos."""
    store = get_store()
    # Business
    acme = store.add_business(
        Business(name="Acme Co.", website="https://acme.example", industry="SaaS", business_type=BusinessType.SAAS)
    )
    # Contacts
    alice = store.add_contact(Contact(email="alice@acme.example", first_name="Alice", last_name="Ng", company="Acme", source="website", tags=["newsletter"]))
    bob = store.add_contact(Contact(email="bob@acme.example", first_name="Bob", last_name="Lee", company="Acme", source="referral", tags=["paid_ads"]))
    # Leads
    l1 = store.add_lead(Lead(title="Website Revamp", contact_id=alice.contact_id, value=15000, source="inbound", pipeline_stage=PipelineStage.INTEREST))
    l2 = store.add_lead(Lead(title="SEO Retainer", contact_id=bob.contact_id, value=3000, source="outbound", pipeline_stage=PipelineStage.CONSIDERATION))
    return {
        "business": acme,
        "contacts": [alice, bob],
        "leads": [l1, l2],
    }
