import streamlit as st
import os
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks
from langchain_groq import ChatGroq
import psutil
import sys
import json
import base64
import logging
import time
from groq import RateLimitError
import re
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import StringIO
import threading
from tenacity import retry, stop_after_attempt, wait_random_exponential
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

try:
    from langsmith import Client
    from langchain.callbacks.tracers.langchain import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    client = Client()
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LangSmith client: {str(e)}")
    client = None

pdf_paths = []

import re
import streamlit as st
from datetime import datetime

def format_log_entry(log_entry):
    timestamp_match = re.search(r'\[(.*?)\]', log_entry)
    timestamp = timestamp_match.group(1) if timestamp_match else "N/A"
    
    action_type_match = re.search(r'\[(DEBUG|INFO|WARNING|ERROR)\]', log_entry)
    action_type = action_type_match.group(1) if action_type_match else None
    
    agent_match = re.search(r'Working Agent: (.*?)(,|\n|$)', log_entry)
    agent_name = agent_match.group(1) if agent_match else None
    
    content = re.sub(r'^\[.*?\](\[.*?\])?', '', log_entry).strip()
    
    return timestamp, action_type, agent_name, content

def display_formatted_log(log_entries):
    for entry in log_entries:
        timestamp, action_type, agent_name, content = format_log_entry(entry)
        
        if action_type:
            st.markdown(f"**:clock1: {timestamp}**")
            st.markdown(f"**Type:** {action_type}")
            if agent_name and agent_name.lower() != "unknown agent":
                st.markdown(f"**Agent:** {agent_name}")
            st.markdown("---")
        
        if content.startswith("Thought:"):
            st.markdown("💭 **Thought:**")
            st.text(content[8:].strip())
        elif content.startswith("Action:"):
            st.markdown("🏃‍♂️ **Action:**")
            st.text(content[7:].strip())
        elif content.startswith("Action Input:"):
            st.markdown("📥 **Action Input:**")
            st.text(content[13:].strip())
        elif content.startswith("Observation:"):
            st.markdown("👁️ **Observation:**")
            st.text(content[12:].strip())
        else:
            st.text(content)
        
        st.markdown("---")

class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
    
    def write(self, data):
        self.buffer.append(data.strip())
        if len(self.buffer) >= 4 or any("Thought:" in line for line in self.buffer):
            with self.expander:
                display_formatted_log(self.buffer)
            self.buffer = []
    
    def flush(self):
        if self.buffer:
            with self.expander:
                display_formatted_log(self.buffer)
            self.buffer = []

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 ** 2  # in MB
    logger.info(f"Memory usage: {mem_usage:.2f} MB")

@st.cache_resource
def get_groq_llm():
    try:
        if client:
            tracer = LangChainTracer()
            callback_manager = CallbackManager([tracer])
        else:
            callback_manager = None
        return ChatGroq(
            temperature=0.2,  # Lower temperature
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            callback_manager=callback_manager
        )
    except Exception as e:
        logger.error(f"Error initializing Groq LLM: {str(e)}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'r') as f:
        data = f.read()
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Function to preprocess the output
def preprocess_output(text):
    # Remove or explain the symbols/numbers
    text = re.sub(r'\[\d+m', '', text)  # Remove color codes
    text = re.sub(r'\[\d+;\d+m', '', text)  # Remove more complex color codes
    return text

# Set page config
st.set_page_config(page_title='Custom Crew AI', layout="centered")

# Custom CSS to make the whole page scrollable and hide the Streamlit container's scrollbar
st.markdown("""
    <style>
    #root > div:nth-child(1) > div > div > div > div > section.main.css-uf99v8.egzxvld5 {
        overflow: visible;
    }
    .stApp {
        overflow-y: auto;
        height: 100vh;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .fadeIn {
        animation: fadeIn 0.5s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# Add the title and description at the top of the app
st.title('📋 ProposalCraft')
st.markdown('Generate comprehensive proposal drafts using AI analysis of your RFP documents.')

# Background section
st.subheader('🎯 Background')
org_name = st.text_input('Please enter the name of the organization or company')
proposal_background = st.text_area('Please provide background on the RFP / proposal that needs to be drafted')

# Total Budget section
st.subheader('💰 Total Budget')
total_budget = st.number_input('Enter the total budget for the proposal:', min_value=0, step=1000, format="%d")

# File upload section
st.subheader("📁 Upload Relevant Documents")
st.markdown("""
Upload PDF files (max 200MB each) that contain:
- Background information about your organization/company
- Details about the core elements of your proposal
- Any relevant RFP (Request for Proposal) documents

The more comprehensive and relevant the uploaded documents the better the AI-generated proposal.
""")

# File uploader widget
uploaded_pdfs = st.file_uploader("Drag and drop files here", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    st.success(f"Successfully uploaded {len(uploaded_pdfs)} file(s).")

if st.button('Draft Proposal'):
    try:
        st.info("Starting the crew. This process may take several minutes depending on the complexity of your documents and requirements. Please wait while our AI agents analyze and generate your proposal draft.")
        
        crew_output_expander = st.expander("Crew Log", expanded=True)
        stream_to_expander = StreamToExpander(crew_output_expander)

        # Redirect stdout to our custom StreamToExpander
        original_stdout = sys.stdout
        sys.stdout = stream_to_expander

        # Process uploaded PDFs
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.getvalue())
                pdf_paths.append(tmp_file.name)
            logger.info(f"Processed: {uploaded_pdf.name}")
        logger.info(f"Total PDFs processed: {len(pdf_paths)}")

        # Initialize agents and tasks
        agents = CustomAgents(pdf_paths)
        tasks = CustomTasks()

        # Set up agents and tasks
        document_ingestion_agent = agents.document_ingestion_agent()
        rfp_analysis_agent = agents.rfp_analysis_agent()
        proposal_writer_agent = agents.proposal_writer_agent()
        budget_specialist_agent = agents.budget_specialist_agent()
        quality_assurance_agent = agents.quality_assurance_agent()

        document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, org_name, proposal_background)
        rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)
        proposal_writing_task = tasks.proposal_writing_task(proposal_writer_agent, "{{rfp_analysis_task.output}}", org_name)
        budget_preparation_task = tasks.budget_preparation_task(budget_specialist_agent, "{{proposal_writing_task.output}}", total_budget)
        quality_review_task = tasks.quality_review_task(quality_assurance_agent, "{{proposal_writing_task.output}}", "{{budget_preparation_task.output}}")

        # Create crew
        crew = Crew(
            agents=[document_ingestion_agent, rfp_analysis_agent, proposal_writer_agent, budget_specialist_agent, quality_assurance_agent],
            tasks=[document_ingestion_task, rfp_analysis_task, proposal_writing_task, budget_preparation_task, quality_review_task],
            verbose=True
        )

        # Run the crew
        with st.spinner("CrewAI Job in Progress..."):
            result = crew.kickoff()

        # Reset stdout
        sys.stdout = original_stdout
        stream_to_expander.flush()  # Flush any remaining content

        st.success("CrewAI Job Completed!")

        # Display final results
        st.subheader("📄 Final Proposal Draft")
        st.markdown(result, unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="Download Proposal Draft",
            data=result,
            file_name="proposal_draft.md",
            mime="text/markdown"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the logs for more details and try again.")
    
    finally:
        # Clean up temporary files
        for path in pdf_paths:
            try:
                os.remove(path)
                logger.info(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {path}: {str(e)}")
        
        log_memory_usage()

# Main execution
if __name__ == "__main__":
    try:
        # Your main app code goes here
        pass  # Remove this if you have actual code to run
    except KeyboardInterrupt:
        st.write("Shutting down gracefully...")
    finally:
        st.stop()

# Create a placeholder for logs
log_placeholder = st.empty()
