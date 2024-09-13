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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

try:
    from langsmith import Client
    from langchain.callbacks.tracers.langchain import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    client = Client()
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LangSmith client: {str(e)}")
    client = None

class StreamToSt:
    def __init__(self, st_component):
        self.st_component = st_component
        self.buffer = ""
        self.current_section = None
        self.content_buffer = []

    def write(self, content):
        self.buffer += content
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                self.process_line(line)
            self.buffer = lines[-1]

    def process_line(self, line):
        if any(keyword in line for keyword in ["Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:"]):
            self.flush_content()
            self.current_section = line.split(':')[0]
            formatted = self.format_output(line)
            self.st_component.markdown(f'<div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">{formatted}', unsafe_allow_html=True)
        elif self.current_section:
            self.content_buffer.append(line.strip())

    def flush_content(self):
        if self.content_buffer:
            content = " ".join(self.content_buffer)
            self.st_component.markdown(content, unsafe_allow_html=True)
            self.content_buffer = []
        if self.current_section:
            self.st_component.markdown("</div>", unsafe_allow_html=True)

    def format_output(self, content):
        if "Thought:" in content:
            return f"🤔 **Thought:** {content.split('Thought:')[1].strip()}"
        elif "Action:" in content:
            return f"🛠️ **Action:** {content.split('Action:')[1].strip()}"
        elif "Action Input:" in content:
            return f"📥 **Action Input:** {content.split('Action Input:')[1].strip()}"
        elif "Observation:" in content:
            return f"👁️ **Observation:** {content.split('Observation:')[1].strip()}"
        elif "Final Answer:" in content:
            return f"🎯 **Final Answer:** {content.split('Final Answer:')[1].strip()}"
        else:
            return content.strip()

    def flush(self):
        if self.buffer:
            self.process_line(self.buffer)
        self.flush_content()
        self.buffer = ""
        self.current_section = None

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
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            callback_manager=callback_manager
        )
    except Exception as e:
        logger.error(f"Error initializing Groq LLM: {str(e)}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(page_title='Custom Crew AI', layout="centered")
st.title('📋 RFP / Proposal Draft')

st.markdown("## 🎯 Background")
org_name = st.text_input('Please enter the name of the organization or company')
proposal_background = st.text_area('Please provide background on the RFP / proposal that needs to be drafted', height=300)

st.markdown("## 📎 Uploaded Files")
uploaded_pdfs = st.file_uploader("Upload PDF files (max 200MB each)", type="pdf", accept_multiple_files=True)
for pdf in uploaded_pdfs:
    st.write(f"• {pdf.name} ({pdf.size / 1024:.1f}KB)")

st.markdown("## 📤 File Upload Instructions")
st.markdown("""
• Drag and drop files here
• Limit 200MB per file
• PDF format
""")

if st.button('Run Custom Crew'):
    logger.info("Run Custom Crew button clicked")
    
    if not org_name or not proposal_background:
        st.error("Please enter both the organization name and background information on the RFP / proposal.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        logger.info(f"Number of PDFs uploaded: {len(uploaded_pdfs)}")
        
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    pdf_paths.append(tmp_file.name)
                logger.info(f"Processed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_pdf.name}: {str(e)}")
        
        logger.info(f"Total PDFs processed: {len(pdf_paths)}")
        
        if pdf_paths:
            try:
                logger.info("Initializing CustomAgents")
                agents = CustomAgents(pdf_paths)
                logger.info("Initializing CustomTasks")
                tasks = CustomTasks()
                
                logger.info("Setting up agents")
                document_ingestion_agent = agents.document_ingestion_agent()
                rfp_analysis_agent = agents.rfp_analysis_agent()
                
                logger.info("Setting up tasks")
                document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, org_name, proposal_background)
                rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)

                logger.info("Creating Crew")
                crew = Crew(
                    agents=[document_ingestion_agent, rfp_analysis_agent],
                    tasks=[document_ingestion_task, rfp_analysis_task],
                    process=Process.sequential,
                    manager_llm=get_groq_llm(),
                )

                logger.info("Starting Crew kickoff")
                
                st.markdown("## 🔄 Status: Crew is working on your proposal...")
                
                # Create a container for the crew's output
                crew_output_container = st.container()

                # Redirect stdout to StreamToSt
                original_stdout = sys.stdout
                stream_to_st = StreamToSt(crew_output_container)
                sys.stdout = stream_to_st

                try:
                    result = crew.kickoff()
                finally:
                    # Restore the original stdout and flush any remaining output
                    sys.stdout = original_stdout
                    stream_to_st.flush()

                # Display the final result
                st.markdown("## 📊 Analysis Result:")
                st.write(result)

                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp_file:
                    json.dump(result, tmp_file, indent=2)
                    tmp_file_name = tmp_file.name

                st.markdown(get_binary_file_downloader_html(tmp_file_name, 'Crew Result'), unsafe_allow_html=True)

                os.remove(tmp_file_name)
            
            except Exception as e:
                error_msg = f"An error occurred during crew execution: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
            
            finally:
                for path in pdf_paths:
                    try:
                        os.remove(path)
                        logger.info(f"Removed temporary file: {path}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file {path}: {str(e)}")
        else:
            st.error("No PDF files were successfully processed. Please try uploading them again.")

log_memory_usage()
