
import streamlit as st
import os
import re
import sys
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks

from langchain_openai import ChatOpenAI
OpenAIGPT4mini = ChatOpenAI(model_name="gpt-4o-mini")
class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)
        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

# Set page config
st.set_page_config(
    page_title='Custom Crew AI',
    layout="centered"
)

st.title('Custom Crew AI Autonomous Grant Proposal System')

# User inputs
var1 = st.text_input('Enter variable 1')
var2 = st.text_input('Enter variable 2')

# File uploader for multiple PDFs
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button('Run Custom Crew'):
    if not var1 or not var2:
        st.error("Please enter both variable 1 and variable 2.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        # Save uploaded PDFs to temporary files
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                pdf_paths.append(tmp_file.name)
        
        # Initialize agents with PDF paths
        agents = CustomAgents(pdf_paths)
        tasks = CustomTasks()
        
        # Set up agents and tasks
        document_ingestion_agent = agents.document_ingestion_agent()
        rfp_analysis_agent = agents.rfp_analysis_agent()
        mission_vision_agent = agents.mission_vision_agent()
        impact_research_agent = agents.impact_research_agent()
        budget_analysis_agent = agents.budget_analysis_agent()
        team_governance_agent = agents.team_governance_agent()
        case_testimonial_agent = agents.case_testimonial_agent()
        quality_integration_agent = agents.quality_integration_agent()
        formatting_submission_agent = agents.formatting_submission_agent()
        project_manager_agent = agents.project_manager_agent()

        document_ingestion_task = tasks.document_ingestion_task(document_ingestion_agent, var1, var2)
        rfp_analysis_task = tasks.rfp_analysis_task(rfp_analysis_agent)
        mission_vision_task = tasks.mission_vision_task(mission_vision_agent)
        impact_research_task = tasks.impact_research_task(impact_research_agent)
        budget_analysis_task = tasks.budget_analysis_task(budget_analysis_agent)
        team_governance_task = tasks.team_governance_task(team_governance_agent)
        case_testimonial_task = tasks.case_testimonial_task(case_testimonial_agent)
        quality_integration_task = tasks.quality_integration_task(quality_integration_agent)
        formatting_submission_task = tasks.formatting_submission_task(formatting_submission_agent)
        project_manager_task = tasks.project_manager_task(project_manager_agent)

        crew = Crew(
            agents=[
                document_ingestion_agent,
                rfp_analysis_agent,
                mission_vision_agent,
                impact_research_agent,
                budget_analysis_agent,
                team_governance_agent,
                case_testimonial_agent,
                quality_integration_agent,
                formatting_submission_agent,
                project_manager_agent
            ],
            tasks=[
                document_ingestion_task,
                rfp_analysis_task,
                mission_vision_task,
                impact_research_task,
                budget_analysis_task,
                team_governance_task,
                case_testimonial_task,
                quality_integration_task,
                formatting_submission_task,
                project_manager_task,
                
            ],
            process=Process.hierarchical,
            manager_llm=OpenAIGPT4mini,
        )

        expander = st.expander("Crew Log")
        sys.stdout = StreamToExpander(expander)
        
        try:
            crew.kickoff()
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            sys.stdout = sys.__stdout__
