import streamlit as st
import os
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks
from langchain_groq import ChatGroq

# Initialize session state
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

def add_debug_message(message):
    st.session_state.debug_messages.append(message)

# Initializations
try:
    groq_llm = ChatGroq(temperature=0,
                model_name="llama3-70b-8192",
                api_key=os.getenv("GROQ_API_KEY"))
    add_debug_message("Groq LLM initialized successfully")
except Exception as e:
    add_debug_message(f"Error initializing Groq LLM: {str(e)}")

st.set_page_config(page_title='Custom Crew AI', layout="centered")
st.title('Custom Crew AI Autonomous Grant Proposal System')

var1 = st.text_input('Please enter the name of the organization or company')
var2 = st.text_area('Please provide background on the RFP / proposal that needs to be drafted', height=300)

uploaded_pdfs = st.file_uploader("Upload PDF files (max 200MB each)", type="pdf", accept_multiple_files=True)

if st.button('Run Custom Crew'):
    add_debug_message("Run Custom Crew button clicked")
    if not var1 or not var2:
        st.error("Please enter both the organization name and background information on the RFP / proposal.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    pdf_paths.append(tmp_file.name)
                add_debug_message(f"Successfully processed {uploaded_pdf.name}")
            except Exception as e:
                add_debug_message(f"Error processing {uploaded_pdf.name}: {str(e)}")
                st.error(f"Error processing {uploaded_pdf.name}: {str(e)}")
                break
        
        if len(pdf_paths) == len(uploaded_pdfs):
            try:
                add_debug_message("Initializing CustomAgents")
                agents = CustomAgents(pdf_paths)
                add_debug_message("Initializing CustomTasks")
                tasks = CustomTasks()
                
                add_debug_message("Setting up agents")
                document_ingestion_agent = agents.document_ingestion_agent()
                rfp_analysis_agent = agents.rfp_analysis_agent()
                
                add_debug_message("Setting up tasks")
                document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, var1, var2)
                rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)

                add_debug_message("Creating Crew")
                crew = Crew(
                    agents=[document_ingestion_agent, rfp_analysis_agent],
                    tasks=[document_ingestion_task, rfp_analysis_task],
                    process=Process.sequential,
                    manager_llm=groq_llm,
                )

                add_debug_message("Starting Crew kickoff")
                result = crew.kickoff()
                add_debug_message("Crew kickoff completed")
                
                st.success("Crew has completed its tasks successfully!")
                st.write(result)
            except Exception as e:
                error_msg = f"An error occurred during crew execution: {str(e)}"
                add_debug_message(error_msg)
                st.error(error_msg)
        else:
            st.error("Not all files were processed successfully. Please try again.")

# Display debug messages
st.subheader("Debug Messages")
for msg in st.session_state.debug_messages:
    st.text(msg)
