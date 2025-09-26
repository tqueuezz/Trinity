"""
Streamlit UI Components Module
Contains all reusable UI components
"""
import streamlit as st
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# CRM imports (lazy optional)
try:
    from crm.crm_models import Contact, Lead, PipelineStage
    from crm.crm_utils import (
        add_contact,
        update_contact,
        find_contacts,
        add_lead,
        update_lead,
        find_leads,
        seed_demo_data,
    )
except Exception:
    Contact = Lead = PipelineStage = object  # type: ignore


def display_header():
    """Display application header"""
    st.markdown(
        """
    <div class="main-header">
        🧬 DeepCode
        OPEN-SOURCE CODE AGENT
        ⚡ DATA INTELLIGENCE LAB @ HKU • REVOLUTIONIZING RESEARCH REPRODUCIBILITY ⚡
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_features():
    """Display DeepCode AI Agent capabilities"""
    # Keep existing showcase minimal to reduce noise when CRM is open
    st.info("DeepCode multi-agent capabilities ready. Switch to CRM to manage leads and contacts.")


def display_status(message: str, status_type: str = "info"):
    """
    Display status message
    Args:
        message: Status message
        status_type: Status type (success, error, warning, info)
    """
    status_classes = {
        "success": "status-success",
        "error": "status-error",
        "warning": "status-warning",
        "info": "status-info",
    }
    icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    css_class = status_classes.get(status_type, "status-info")
    icon = icons.get(status_type, "ℹ️")
    st.markdown(
        f"""
    <div class="{css_class}">
        {icon} {message}
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------- CRM Components (Scaffold) ----------------

def crm_contact_form(defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Basic contact create/edit form"""
    defaults = defaults or {}
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name", value=defaults.get("first_name", ""))
        email = st.text_input("Email", value=defaults.get("email", ""))
        source = st.text_input("Source", value=defaults.get("source", "website"))
    with col2:
        last_name = st.text_input("Last Name", value=defaults.get("last_name", ""))
        phone = st.text_input("Phone", value=defaults.get("phone", ""))
        company = st.text_input("Company", value=defaults.get("company", ""))
    tags = st.text_input("Tags (comma-separated)", value=",").strip()
    notes = st.text_area("Notes", value=defaults.get("notes", ""), height=100)
    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "company": company,
        "source": source,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "notes": notes,
    }


def crm_lead_form(defaults: Dict[str, Any] | None = None, contacts: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Basic lead create/edit form"""
    defaults = defaults or {}
    contact_options = [
        (c.get("contact_id", ""), f"{c.get('first_name','')} {c.get('last_name','')} - {c.get('email','')}") for c in (contacts or [])
    ]
    title = st.text_input("Lead Title", value=defaults.get("title", ""))
    description = st.text_area("Description", value=defaults.get("description", ""))
    value = st.number_input("Estimated Value ($)", value=float(defaults.get("value", 0.0)), min_value=0.0, step=100.0)
    source = st.text_input("Source", value=defaults.get("source", ""))
    contact_choice = st.selectbox("Contact", options=[opt[1] for opt in contact_options] or [""], index=0)
    contact_id = ""
    if contact_options and contact_choice:
        contact_id = contact_options[[o[1] for o in contact_options].index(contact_choice)][0]
    stage = st.selectbox(
        "Pipeline Stage",
        options=[s.value for s in PipelineStage] if hasattr(PipelineStage, "__iter__") else ["awareness", "interest", "consideration"],
        index=0,
    )
    return {
        "title": title,
        "description": description,
        "value": value,
        "source": source,
        "contact_id": contact_id,
        "pipeline_stage": stage,
    }


def crm_contacts_table(rows: List[Dict[str, Any]]):
    """Display contacts in a simple table (pandas optional)"""
    try:
        import pandas as pd
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No contacts yet.")
    except Exception:
        for r in rows:
            st.write(f"- {r.get('first_name','')} {r.get('last_name','')} | {r.get('email','')} | {r.get('source','')}")


def crm_leads_table(rows: List[Dict[str, Any]]):
    """Display leads in a simple table (pandas optional)"""
    try:
        import pandas as pd
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No leads yet.")
    except Exception:
        for r in rows:
            st.write(f"- {r.get('title','')} | ${r.get('value',0)} | {r.get('pipeline_stage','')}")


def crm_pipeline_stage_list() -> List[str]:
    try:
        return [s.value for s in PipelineStage]
    except Exception:
        return ["awareness", "interest", "consideration", "intent", "evaluation", "purchase", "retention", "advocacy"]


def crm_dashboard_component():
    """Minimal CRM dashboard scaffold"""
    st.subheader("CRM Dashboard")

    # Seed demo button
    if st.button("Seed Demo Data", use_container_width=True):
        try:
            seed_demo_data()
            st.success("Seeded demo data.")
        except Exception as e:
            st.warning(f"Could not seed demo data: {e}")

    # Search contacts
    st.markdown("### Contacts")
    q = st.text_input("Search contacts", value="")
    filters = {}
    results = []
    try:
        results = find_contacts(query=q, **filters)
    except Exception:
        pass
    # Normalize for display
    contact_rows = []
    for c in results:
        if hasattr(c, "__dict__"):
            row = {**c.__dict__}
            row.pop("custom_data", None)
            contact_rows.append(row)
    crm_contacts_table(contact_rows)

    with st.expander("Add New Contact"):
        form_vals = crm_contact_form()
        if st.button("Create Contact"):
            try:
                created = add_contact(**form_vals)
                st.success(f"Contact created: {getattr(created, 'email', '')}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to create contact: {e}")

    # Leads section
    st.markdown("### Leads")
    try:
        leads = find_leads()
    except Exception:
        leads = []
    lead_rows = []
    for l in leads:
        if hasattr(l, "__dict__"):
            row = {**l.__dict__}
            row.pop("custom_data", None)
            lead_rows.append(row)
    crm_leads_table(lead_rows)

    with st.expander("Add New Lead"):
        form_vals = crm_lead_form(
            contacts=[r for r in contact_rows],
        )
        if st.button("Create Lead"):
            try:
                created = add_lead(**form_vals)
                st.success(f"Lead created: {getattr(created, 'title', '')}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to create lead: {e}")


# Keep other existing components (status, progress, etc.) minimal placeholders to avoid duplication

