import streamlit as st


st.set_page_config(page_title="Lincolnshire Mental Health Dashboard", layout="wide")

st.title("Lincolnshire Mental Health Dashboard")
st.markdown(
    """
This dashboard contains:

- Lincolnshire Mental Health Need Index
- Small Area Mental Health Index (SAMHI) 
- ComparisonBetween Need Index and SAMHI

Use the page selector in the sidebar to navigate.
"""
)
