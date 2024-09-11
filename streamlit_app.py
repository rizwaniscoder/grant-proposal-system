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
import re

class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []

    def write(self, data):
        self.buffer.append(data)
        formatted_output = self.format_output(''.join(self.buffer))
        self.expander.markdown(formatted_output, unsafe_allow_html=True)

    def flush(self):
        pass

    def format_output(self, text):
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('Thought:'):
                formatted_lines.append(f'<span style="color: blue;">{line}</span>')
            elif line.startswith('Action:'):
                formatted_lines.append(f'<span style="color: green;">{line}</span>')
            elif line.startswith('Action Input:'):
                formatted_lines.append(f'<span style="color: orange;">{line}</span>')
            elif line.startswith('Observation:'):
                formatted_lines.append(f'<span style="color: red;">{line}</span>')
            elif line.startswith('Human:') or line.startswith('AI:'):
                formatted_lines.append(f'<strong>{line}</strong>')
            else:
                formatted_lines.append(line)
        
        return '<br>'.join(formatted_lines)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 ** 2  # in MB
    st.sidebar.text(f"Memory usage: {mem_usage:.2f} MB")

@st.cache_resource
def get_groq_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

def add_debug_message(message):
    st.session_state.debug_messages.append(message)
    log_memory_usage()

st.set_page_config(page_title='Custom Crew AI', layout="centered")
st.title('Custom Crew AI Autonomous Grant Proposal System')

org_name = st.text_input('Please enter the name of the organization or company')
proposal_background = st.text_area('Please provide background on the RFP / proposal that needs to be drafted', height=300)

uploaded_pdfs = st.file_uploader("Upload PDF files (max 200MB each)", type="pdf", accept_multiple_files=True)

if st.button('Run Custom Crew'):
    add_debug_message("Run Custom Crew button clicked")
    
    if not org_name or not proposal_background:
        st.error("Please enter both the organization name and background information on the RFP / proposal.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        add_debug_message(f"Number of PDFs uploaded: {len(uploaded_pdfs)}")
        
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    pdf_paths.append(tmp_file.name)
                add_debug_message(f"Processed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_pdf.name}: {str(e)}")
        
        add_debug_message(f"Total PDFs processed: {len(pdf_paths)}")
        
        if pdf_paths:
            try:
                add_debug_message("Initializing CustomAgents")
                agents = CustomAgents(pdf_paths)
                add_debug_message("Initializing CustomTasks")
                tasks = CustomTasks()
                
                add_debug_message("Setting up agents")
                document_ingestion_agent = agents.document_ingestion_agent()
                rfp_analysis_agent = agents.rfp_analysis_agent()
                
                add_debug_message("Setting up tasks")
                document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, org_name, proposal_background)
                rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)

                add_debug_message("Creating Crew")
                crew = Crew(
                    agents=[document_ingestion_agent, rfp_analysis_agent],
                    tasks=[document_ingestion_task, rfp_analysis_task],
                    process=Process.sequential,
                    manager_llm=get_groq_llm(),
                )

                add_debug_message("Starting Crew kickoff")
                
                # Create an expander for CrewAI logs
                crew_log_expander = st.expander("CrewAI Logs", expanded=True)
                
                # Redirect stdout to the expander
                original_stdout = sys.stdout
                sys.stdout = StreamToExpander(crew_log_expander)
                
                with st.spinner('Crew is working on your proposal...'):
                    result = crew.kickoff()
                
                # Restore original stdout
                sys.stdout = original_stdout
                
                st.success("Crew has completed its tasks successfully!")
                
                # Display the full result
                st.subheader("Crew Result:")
                st.json(result)

                # Provide download link for the result
                if result:
                    # Save result to a file
                    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp_file:
                        json.dump(result, tmp_file, indent=2)
                        tmp_file_name = tmp_file.name

                    # Provide download link
                    st.markdown(get_binary_file_downloader_html(tmp_file_name, 'Crew Result'), unsafe_allow_html=True)

                    # Clean up the temporary file
                    os.remove(tmp_file_name)
            
            except Exception as e:
                error_msg = f"An error occurred during crew execution: {str(e)}"
                add_debug_message(error_msg)
                st.error(error_msg)
            
            finally:
                # Clean up temporary files
                for path in pdf_paths:
                    try:
                        os.remove(path)
                        add_debug_message(f"Removed temporary file: {path}")
                    except Exception as e:
                        add_debug_message(f"Error removing temporary file {path}: {str(e)}")
        else:
            st.error("No PDF files were successfully processed. Please try uploading them again.")

# Display debug messages
st.subheader("Debug Messages")
for msg in st.session_state.debug_messages:
    st.text(msg)

log_memory_usage()
