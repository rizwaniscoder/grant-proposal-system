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
            self.st_component.markdown(f'<div class="fadeIn" style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">{formatted}</div>', unsafe_allow_html=True)
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
            return f'<div style="background-color: #f0f0f0; padding: 5px; border-radius: 3px;">🤔 <strong style="color: #2c3e50;">Thought:</strong></div> {content.split("Thought:")[1].strip()}'
        elif "Action:" in content:
            return f'<div style="background-color: #e6f3ff; padding: 5px; border-radius: 3px;">🛠️ <strong style="color: #3498db;">Action:</strong></div> {content.split("Action:")[1].strip()}'
        elif "Action Input:" in content:
            return f'<div style="background-color: #fff5e6; padding: 5px; border-radius: 3px;">📥 <strong style="color: #e67e22;">Action Input:</strong></div> {content.split("Action Input:")[1].strip()}'
        elif "Observation:" in content:
            return f'<div style="background-color: #e6ffe6; padding: 5px; border-radius: 3px;">👁️ <strong style="color: #27ae60;">Observation:</strong></div> {content.split("Observation:")[1].strip()}'
        elif "Final Answer:" in content:
            answer = content.split("Final Answer:")[1].strip()
            return f'<div style="background-color: #ffe6e6; padding: 5px; border-radius: 3px;">🎯 <strong style="color: #e74c3c;">Final Answer:</strong></div>\n\n{answer}'
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
    with open(bin_file, 'r') as f:
        data = f.read()
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Set page config
st.set_page_config(page_title='Custom Crew AI', layout="centered")

# Custom CSS to control width
st.markdown("""
    <style>
        .reportview-container .main .block-container{
            max-width: 800px;
            padding-top: 5rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title('📋 RFP / Proposal Draft')

st.markdown("## Background")
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
                proposal_writer_agent = agents.proposal_writer_agent()
                budget_specialist_agent = agents.budget_specialist_agent()
                quality_assurance_agent = agents.quality_assurance_agent()
                
                logger.info("Setting up tasks")
                document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, org_name, proposal_background)
                rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)
                proposal_writing_task = tasks.proposal_writing_task(proposal_writer_agent, "{{rfp_analysis_task.output}}", org_name)
                budget_preparation_task = tasks.budget_preparation_task(budget_specialist_agent, "{{proposal_writing_task.output}}")
                quality_review_task = tasks.quality_review_task(quality_assurance_agent, "{{proposal_writing_task.output}}\n{{budget_preparation_task.output}}")

                logger.info("Creating Crew")
                crew = Crew(
                agents=[document_ingestion_agent, rfp_analysis_agent, proposal_writer_agent, budget_specialist_agent, quality_assurance_agent],
                tasks=[document_ingestion_task, rfp_analysis_task, proposal_writing_task, budget_preparation_task, quality_review_task],
                process=Process.sequential,
                manager_llm=get_groq_llm(),
                verbose=True,  # Change this from 2 to True
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
                    for task in crew.tasks:
                        task_result = task.execute()
                        st.markdown(f"## Task Completed: {task.name}")
                        st.write(task_result)
                        if st.button("Continue to next task"):
                            continue
                        else:
                            st.stop()
                    result = crew.get_final_output()
                finally:
                    # Restore the original stdout and flush any remaining output
                    sys.stdout = original_stdout
                    stream_to_st.flush()

                # Display the final result
                st.markdown("## 📊 Analysis Result:")
                
                # Convert CrewOutput to a dictionary
                result_dict = {
                    "final_output": result.final_output,
                    "tasks": [
                        {
                            "task_name": task.name,
                            "output": task.output
                        } for task in result.tasks
                    ]
                }
                
                st.json(json.dumps(result_dict, indent=2))

                # Save the result as a JSON file
                with open("proposal_result.json", "w") as f:
                    json.dump(result_dict, f, indent=2)

                # Provide a download link for the JSON file
                st.markdown(get_binary_file_downloader_html("proposal_result.json", "Download Proposal Result"), unsafe_allow_html=True)

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
