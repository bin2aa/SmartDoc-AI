"""Reusable UI components for SmartDoc AI."""

import streamlit as st
from typing import Optional, List


class UIComponents:
    """Reusable UI components following Component Pattern."""
    
    @staticmethod
    def file_uploader(
        label: str = "Upload Document",
        accepted_types: List[str] = None,
        accept_multiple_files: bool = False,
    ):
        """
        Reusable file uploader component.
        
        Args:
            label: Label for the file uploader
            accepted_types: List of accepted file extensions
            
        Returns:
            Uploaded file object or None
        """
        if accepted_types is None:
            accepted_types = ['pdf', 'docx', 'txt']
        
        return st.file_uploader(
            label,
            type=accepted_types,
            accept_multiple_files=accept_multiple_files,
            help=f"Supported formats: {', '.join(accepted_types)}"
        )
    
    @staticmethod
    def chat_message(role: str, content: str, avatar: str = None):
        """
        Render a chat message bubble.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            avatar: Optional avatar emoji
        """
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
    
    @staticmethod
    def loading_spinner(message: str = "Processing..."):
        """
        Context manager for loading spinner.
        
        Args:
            message: Loading message
            
        Returns:
            Spinner context manager
        """
        return st.spinner(message)
    
    @staticmethod
    def error_alert(message: str, details: Optional[str] = None):
        """
        Styled error alert.
        
        Args:
            message: Error message
            details: Optional error details
        """
        st.error(f"❌ {message}")
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def success_alert(message: str):
        """
        Styled success alert.
        
        Args:
            message: Success message
        """
        st.success(f"✅ {message}")
    
    @staticmethod
    def info_alert(message: str):
        """
        Styled info alert.
        
        Args:
            message: Info message
        """
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def warning_alert(message: str):
        """
        Styled warning alert.
        
        Args:
            message: Warning message
        """
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def sidebar_section(title: str, icon: str = "📌"):
        """
        Create sidebar section with title.
        
        Args:
            title: Section title
            icon: Section icon
        """
        st.sidebar.markdown(f"### {icon} {title}")
    
    @staticmethod
    def metric_card(label: str, value: str, delta: Optional[str] = None):
        """
        Display a metric card.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta value
        """
        st.metric(label=label, value=value, delta=delta)
