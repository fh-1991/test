import streamlit as st

def initialize_session_state():
    """セッションステートの初期化"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'target_case' not in st.session_state:
        st.session_state.target_case = None
    
    if 'weights_updated' not in st.session_state:
        st.session_state.weights_updated = False
