"""CRM Data Models for Trinity Strategy & Marketing

This module contains the core data models for the CRM system, designed to be
flexible enough for marketing automation services while maintaining the ability
to adapt to different business types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import json


class ContactStatus(Enum):
    """Contact status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BLOCKED = "blocked"
    PENDING = "pending"


class LeadStatus(Enum):
    """Lead status enumeration"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    NURTURING = "nurturing"


class PipelineStage(Enum):
    """Sales pipeline stages for marketing services"""
    AWARENESS = "awareness"
    INTEREST = "interest"
    CONSIDERATION = "consideration"
    INTENT = "intent"
    EVALUATION = "evaluation"
    PURCHASE = "purchase"
    RETENTION = "retention"
    ADVOCACY = "advocacy"


class BusinessType(Enum):
    """Business type classification"""
    MARKETING_AGENCY = "marketing_agency"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    CONSULTING = "consulting"
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    REAL_ESTATE = "real_estate"
    EDUCATION = "education"
    OTHER = "other"


@dataclass
class Contact:
    """Core contact model for CRM system"""
    
    # Basic identification
    contact_id: Optional[str] = None
    email: str = ""
    phone: str = ""
    first_name: str = ""
    last_name: str = ""
    company: str = ""
    
    # Contact metadata
    source: str = ""  # Where did they come from (website, referral, ad, etc.)
    status: ContactStatus = ContactStatus.ACTIVE
    
    # Marketing automation fields
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Flexible data field for business-specific information
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_contacted: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    @property
    def full_name(self) -> str:
        """Get full name of contact"""
        return f"{self.first_name} {self.last_name}".strip()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the contact"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the contact"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def update_custom_data(self, key: str, value: Any) -> None:
        """Update custom data field"""
        self.custom_data[key] = value
        self.updated_at = datetime.now()


@dataclass
class Lead:
    """Lead model for sales pipeline management"""
    
    # Basic identification
    lead_id: Optional[str] = None
    contact_id: Optional[str] = None  # Reference to associated contact
    
    # Lead details
    title: str = ""
    description: str = ""
    value: float = 0.0  # Estimated deal value
    probability: float = 0.0  # Win probability (0-100)
    
    # Pipeline management
    status: LeadStatus = LeadStatus.NEW
    pipeline_stage: PipelineStage = PipelineStage.AWARENESS
    
    # Activity tracking
    scheduled_activity: Optional[Dict[str, Any]] = None  # Next scheduled activity
    activities: List[Dict[str, Any]] = field(default_factory=list)  # Activity history
    
    # Marketing fields
    source: str = ""
    campaign: str = ""
    
    # Flexible data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expected_close_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def add_activity(self, activity_type: str, description: str, 
                    scheduled_for: Optional[datetime] = None) -> None:
        """Add an activity to the lead"""
        activity = {
            "type": activity_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "scheduled_for": scheduled_for.isoformat() if scheduled_for else None,
            "completed": False
        }
        self.activities.append(activity)
        self.updated_at = datetime.now()
    
    def schedule_next_activity(self, activity_type: str, description: str, 
                              scheduled_for: datetime) -> None:
        """Schedule the next activity"""
        self.scheduled_activity = {
            "type": activity_type,
            "description": description,
            "scheduled_for": scheduled_for.isoformat()
        }
        self.updated_at = datetime.now()
    
    def advance_pipeline_stage(self) -> bool:
        """Advance to next pipeline stage"""
        stages = list(PipelineStage)
        current_index = stages.index(self.pipeline_stage)
        
        if current_index < len(stages) - 1:
            self.pipeline_stage = stages[current_index + 1]
            self.updated_at = datetime.now()
            return True
        return False


@dataclass
class Business:
    """Business/Company model for client management"""
    
    # Basic information
    business_id: Optional[str] = None
    name: str = ""
    website: str = ""
    industry: str = ""
    business_type: BusinessType = BusinessType.OTHER
    
    # Contact information
    primary_email: str = ""
    primary_phone: str = ""
    address: Dict[str, str] = field(default_factory=dict)  # street, city, state, zip, country
    
    # Business details
    employee_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    
    # Relationship management
    client_since: Optional[datetime] = None
    account_manager: str = ""
    
    # Marketing automation specific
    marketing_segments: List[str] = field(default_factory=list)
    automation_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Service tracking
    active_services: List[str] = field(default_factory=list)
    service_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Flexible data for different business types
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def add_service(self, service_name: str, start_date: Optional[datetime] = None) -> None:
        """Add an active service"""
        if service_name not in self.active_services:
            self.active_services.append(service_name)
            
            # Add to service history
            service_record = {
                "service": service_name,
                "start_date": (start_date or datetime.now()).isoformat(),
                "status": "active"
            }
            self.service_history.append(service_record)
            self.updated_at = datetime.now()
    
    def remove_service(self, service_name: str, end_date: Optional[datetime] = None) -> None:
        """Remove an active service"""
        if service_name in self.active_services:
            self.active_services.remove(service_name)
            
            # Update service history
            for record in reversed(self.service_history):
                if record["service"] == service_name and record["status"] == "active":
                    record["end_date"] = (end_date or datetime.now()).isoformat()
                    record["status"] = "completed"
                    break
            
            self.updated_at = datetime.now()
    
    def add_marketing_segment(self, segment: str) -> None:
        """Add business to a marketing segment"""
        if segment not in self.marketing_segments:
            self.marketing_segments.append(segment)
            self.updated_at = datetime.now()
    
    def set_automation_preference(self, key: str, value: Any) -> None:
        """Set automation preference"""
        self.automation_preferences[key] = value
        self.updated_at = datetime.now()


# Utility functions for working with models

def create_marketing_contact(email: str, first_name: str = "", last_name: str = "", 
                            company: str = "", source: str = "website") -> Contact:
    """Create a contact optimized for marketing automation"""
    contact = Contact(
        email=email,
        first_name=first_name,
        last_name=last_name,
        company=company,
        source=source
    )
    # Add default marketing tags
    contact.add_tag("marketing_lead")
    contact.add_tag(f"source_{source}")
    
    return contact


def create_sales_lead(title: str, contact_id: str, value: float = 0.0, 
                     source: str = "") -> Lead:
    """Create a sales lead"""
    return Lead(
        title=title,
        contact_id=contact_id,
        value=value,
        source=source,
        status=LeadStatus.NEW,
        pipeline_stage=PipelineStage.AWARENESS
    )


def create_client_business(name: str, business_type: BusinessType = BusinessType.OTHER,
                          primary_email: str = "", website: str = "") -> Business:
    """Create a business/client record"""
    business = Business(
        name=name,
        business_type=business_type,
        primary_email=primary_email,
        website=website
    )
    
    # Set client relationship
    business.client_since = datetime.now()
    business.add_marketing_segment("active_client")
    
    return business
