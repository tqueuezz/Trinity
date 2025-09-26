"""
Streamlit Page Layout Module
Contains main page layout and flow control with CRM-aware header/sidebar
"""
import streamlit as st
from .components import (
    display_header,
    display_features,
    sidebar_control_panel,
    input_method_selector,
    results_display_component,
    footer_component,
)
from .handlers import (
    initialize_session_state,
    handle_start_processing_button,
    handle_error_display,
)
from .styles import get_main_styles

# New: CRM imports for permissions & business switching (lazy optional)
try:
    from crm.crm_auth import current_business_context, switch_business_context, Role
except Exception:
    current_business_context = lambda s: s.get("active_business_id")
    def switch_business_context(session, user, biz_id):
        session["active_business_id"] = biz_id
        return True
    class Role:  # fallback enum-like
        OWNER = type("T", (), {"value": "owner"})

ASSETS_LOGO_PATH_CANDIDATES = [
    "assets/Deepcode.png",
    "assets/Main Logo S&M Transparent.png",
    "assets/genericlogo.png",
    "assets/logo.png",
]

def setup_page_config():
    """Setup page configuration"""
    st.set_page_config(
        page_title="DeepCode - AI Research Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_styles():
    """Apply custom styles"""
    st.markdown(get_main_styles(), unsafe_allow_html=True)


def _resolve_logo_path() -> str | None:
    for p in ASSETS_LOGO_PATH_CANDIDATES:
        try:
            # Streamlit can show local relative path if exists in repo
            return p
        except Exception:
            continue
    return None


def _crm_header(user=None):
    """Render dynamic header with logo, business switch, and permission-aware nav."""
    cols = st.columns([0.1, 0.5, 0.4])
    with cols[0]:
        logo = _resolve_logo_path()
        if logo:
            st.image(logo, use_container_width=True)
    with cols[1]:
        st.subheader("Trinity CRM Suite")
        active_biz = current_business_context(st.session_state)
        st.caption(f"Business: {active_biz or 'No business selected'}")
    with cols[2]:
        if user and getattr(user, "memberships", None):
            options = [m.business_id for m in user.memberships]
            default = options.index(active_biz) if active_biz in options else 0
            selected = st.selectbox("Switch business", options, index=default)
            if st.button("Apply", key="apply_biz_switch"):
                switch_business_context(st.session_state, user, selected)
        else:
            st.info("Connect or create a business in Onboarding")

    st.markdown("---")


def _crm_sidebar(user=None):
    """Sidebar with permissions-based visibility and SaaS placeholders."""
    with st.sidebar:
        st.markdown("### Navigation")
        active_biz = current_business_context(st.session_state)
        can_view_contacts = bool(user and active_biz and user.has_permission(active_biz, "contacts:read")) if user else True
        can_view_leads = bool(user and active_biz and user.has_permission(active_biz, "leads:read")) if user else True
        can_manage_pipeline = bool(user and active_biz and user.has_permission(active_biz, "pipeline:read")) if user else True
        can_admin = bool(user and active_biz and user.has_permission(active_biz, "settings:read")) if user else True

        if can_view_contacts:
            st.page_link("/contacts", label="Contacts", icon="👤")
        if can_view_leads:
            st.page_link("/leads", label="Leads", icon="📈")
        if can_manage_pipeline:
            st.page_link("/pipeline", label="Pipeline", icon="🪜")

        st.markdown("---")
        st.markdown("### Admin & Onboarding")
        # SaaS onboarding admin placeholder
        if can_admin:
            st.page_link("/admin/onboarding", label="Onboarding Admin", icon="🧭")
            st.page_link("/admin/business-wizard", label="Business Wizard", icon="🧩")

        st.markdown("---")
        return sidebar_control_panel()


def render_main_content():
    """Render main content area"""
    # Dynamic CRM header
    user = st.session_state.get("__user__")  # integration point for auth layer
    _crm_header(user)

    # Display original header and features
    display_header()
    display_features()
    st.markdown("---")

    # Display results if available
    if st.session_state.show_results and st.session_state.last_result:
        results_display_component(
            st.session_state.last_result, st.session_state.task_counter
        )
        st.markdown("---")
        return

    # Show input interface only when not displaying results
    if not st.session_state.show_results:
        render_input_interface()

    # Display error messages if any
    handle_error_display()


def render_input_interface():
    """Render input interface"""
    # Get input source and type
    input_source, input_type = input_method_selector(st.session_state.task_counter)

    # Processing button
    if input_source and not st.session_state.processing:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            handle_start_processing_button(input_source, input_type)
    elif st.session_state.processing:
        st.info("🔄 Processing in progress... Please wait.")
        st.warning("⚠️ Do not refresh the page or close the browser during processing.")
    elif not input_source:
        st.info("👆 Please upload a file or enter a URL to start processing.")


def render_sidebar():
    """Render sidebar with CRM navigation"""
    user = st.session_state.get("__user__")
    return _crm_sidebar(user)


def main_layout():
    """Main layout function"""
    # Initialize session state
    initialize_session_state()

    # Setup page configuration
    setup_page_config()

    # Apply custom styles
    apply_custom_styles()

    # Render sidebar
    sidebar_info = render_sidebar()

    # Render main content
    render_main_content()

    # Display footer
    footer_component()

    return sidebar_info
