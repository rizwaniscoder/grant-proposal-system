from crewai import Agent
from langchain.tools import Tool
from langchain_groq import ChatGroq as Groq
import os

class CustomAgents:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def document_ingestion_agent(self):
        return Agent(
            role='Expert Document Analyzer',
            goal='Thoroughly analyze and extract key information from uploaded documents',
            backstory='You are an expert in document analysis with a keen eye for detail.',
            tools=[],  # Empty list if no specific tools are needed
            verbose=True,
            llm=self.get_groq_llm()
        )

    def rfp_analysis_agent(self):
        return Agent(
            role='RFP Analysis Specialist',
            goal='Analyze RFP requirements and create detailed task outlines',
            backstory='You are an expert in breaking down complex RFPs into actionable tasks.',
            tools=[],  # Empty list if no specific tools are needed
            verbose=True,
            llm=self.get_groq_llm()
        )

    def proposal_writer_agent(self):
        return Agent(
            role='Grant Proposal Writer',
            goal='Write compelling and comprehensive grant proposals',
            backstory='You are a skilled writer with extensive experience in crafting successful grant proposals.',
            tools=[],  # Empty list if no specific tools are needed
            verbose=True,
            llm=self.get_groq_llm()
        )

    def budget_specialist_agent(self):
        return Agent(
            role='Nonprofit Budget Specialist',
            goal='Develop comprehensive budgets for grant proposals',
            backstory='You are an expert in creating detailed and realistic budgets for nonprofit organizations.',
            tools=[],  # Empty list if no specific tools are needed
            verbose=True,
            llm=self.get_groq_llm()
        )

    def quality_assurance_agent(self):
        return Agent(
            role='Proposal Quality Assurance Specialist',
            goal='Ensure the final proposal meets all requirements and is of high quality',
            backstory='You have a keen eye for detail and extensive experience in reviewing grant proposals.',
            tools=[],  # Empty list if no specific tools are needed
            verbose=True,
            llm=self.get_groq_llm()
        )

    def get_groq_llm(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return Groq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            api_key=api_key
        )