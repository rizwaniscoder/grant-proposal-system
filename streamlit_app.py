import streamlit as st
import os
import re
import sys
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# ... (previous code remains unchanged)

st.title('Custom Crew AI Autonomous Grant Proposal System')

# User inputs
var1 = st.text_input('Please enter the name of the organization or company')
var2 = st.text_area('Please provide background on the RFP / proposal that needs to be drafted', height=300)

# File uploader for multiple PDFs
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button('Run Custom Crew'):
    if not var1 or not var2:
        st.error("Please enter both the organization name and background information on the RFP / proposal.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        # ... (rest of the code remains unchanged)

# ... (rest of the file remains unchanged)
