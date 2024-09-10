import streamlit as st
import os
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks
from langchain_groq import ChatGroq

# Use st.cache_resource for objects that should persist across reruns
@st.cache_resource
def get_groq_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

# Initialize session state
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = []

def add_debug_message(message):
    st.session_state.debug_messages.append(message)

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
        
        st.session_state.pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    st.session_state.pdf_paths.append(tmp_file.name)
                add_debug_message(f"Processed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_pdf.name}: {str(e)}")
        
        add_debug_message(f"Total PDFs processed: {len(st.session_state.pdf_paths)}")
        
        if st.session_state.pdf_paths:
            try:
                add_debug_message("Initializing CustomAgents")
                agents = CustomAgents(st.session_state.pdf_paths)
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
                with st.spinner('Crew is working on your proposal...'):
                    result = crew.kickoff()
                
                st.success("Crew has completed its tasks successfully!")
                st.write("Crew Result:")
                st.write(result)
            
            except Exception as e:
                error_msg = f"An error occurred during crew execution: {str(e)}"
                add_debug_message(error_msg)
                st.error(error_msg)
        else:
            st.error("No PDF files were successfully processed. Please try uploading them again.")

# Display debug messages
st.subheader("Debug Messages")
for msg in st.session_state.debug_messages:
    st.text(msg)

# Clean up temporary files
for path in st.session_state.pdf_paths:
    try:
        os.remove(path)
        add_debug_message(f"Removed temporary file: {path}")
    except Exception as e:
        add_debug_message(f"Error removing temporary file {path}: {str(e)}")

# Clear pdf_paths after processing
st.session_state.pdf_paths = []
