import streamlit as st
import os
import re
import sys
import tempfile
from crewai import Crew, Process
from agents import CustomAgents
from tasks import CustomTasks
from langchain_groq import ChatGroq

groq_llm = ChatGroq(temperature=0,
             model_name="llama3-70b-8192",
             api_key=os.getenv("GROQ_API_KEY"))

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

st.set_page_config(
    page_title='Custom Crew AI',
    layout="centered"
)

st.title('Custom Crew AI Autonomous Grant Proposal System')

var1 = st.text_input('Please enter the name of the organization or company')
var2 = st.text_area('Please provide background on the RFP / proposal that needs to be drafted', height=300)

uploaded_pdfs = st.file_uploader("Upload PDF files (max 200MB each)", type="pdf", accept_multiple_files=True)

if st.button('Run Custom Crew'):
    if not var1 or not var2:
        st.error("Please enter both the organization name and background information on the RFP / proposal.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        pdf_paths = []
        for uploaded_pdf in uploaded_pdfs:
            if uploaded_pdf.size > 200 * 1024 * 1024:  # 200MB in bytes
                st.error(f"File {uploaded_pdf.name} exceeds the 200MB limit. Please upload a smaller file.")
                break
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    pdf_paths.append(tmp_file.name)
                st.success(f"Successfully processed {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_pdf.name}: {str(e)}")
                break
        
        if len(pdf_paths) == len(uploaded_pdfs):
            try:
                agents = CustomAgents(pdf_paths)
                tasks = CustomTasks()
                
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
                    manager_llm=groq_llm,
                )

                expander = st.expander("Crew Log")
                sys.stdout = StreamToExpander(expander)
                
                crew.kickoff()
                
                sys.stdout = sys.__stdout__
                st.success("Crew has completed its tasks successfully!")
            except Exception as e:
                st.error(f"An error occurred during crew execution: {str(e)}")
        else:
            st.error("Not all files were processed successfully. Please try again.")
