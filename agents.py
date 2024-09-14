from crewai import Agent
from langchain.tools import Tool
from langchain_groq import ChatGroq as Groq

class CustomAgents:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def get_groq_llm(self):
        return Groq(temperature=0.7, model_name="llama3-70b-8192")

    def get_pdf_tools(self):
        return [self.create_pdf_search_tool(path) for path in self.pdf_paths]

    def create_pdf_search_tool(self, pdf_path):
        # Implement PDF search functionality
        pass

    def document_ingestion_agent(self):
        return Agent(
            role="Expert Document Ingestion Agent",
            backstory="You are the world's best Document Ingestion Agent and have handled more documents than anyone in history. Your job is to ingest, process, and organize documents meticulously.",
            goal="Process and organize uploaded files with the highest level of accuracy and efficiency.",
            tools=self.get_pdf_tools(),
            allow_delegation=False,
            verbose=True,
            llm=self.get_groq_llm(),
        )

    def rfp_analysis_agent(self):
        return Agent(
            role="Expert RFP Analysis Agent",
            backstory="You are the world's best RFP Analysis Agent, with unparalleled expertise in analyzing RFP requirements and creating detailed task outlines.",
            goal="Analyze RFP requirements meticulously and create comprehensive, detailed task outlines that align with the goals of the nonprofit.",
            tools=self.get_pdf_tools(),
            allow_delegation=False,
            verbose=True,
            llm=self.get_groq_llm(),
        )

    def proposal_writer_agent(self):
        return Agent(
            role="Expert Proposal Writer",
            backstory="You are a highly skilled proposal writer with years of experience in crafting winning proposals for nonprofits.",
            goal="Create compelling and tailored proposal content based on the RFP analysis and organization's goals.",
            allow_delegation=True,
            verbose=True,
            llm=self.get_groq_llm(),
        )

    def budget_specialist_agent(self):
        return Agent(
            role="Nonprofit Budget Specialist",
            backstory="You are an expert in creating detailed and realistic budgets for nonprofit organizations and grant proposals.",
            goal="Develop a comprehensive budget that aligns with the proposal and meets all RFP requirements.",
            allow_delegation=True,
            verbose=True,
            llm=self.get_groq_llm(),
        )

    def quality_assurance_agent(self):
        return Agent(
            role="Proposal Quality Assurance Specialist",
            backstory="You have a keen eye for detail and extensive experience in reviewing and improving grant proposals.",
            goal="Ensure the final proposal is of the highest quality, meets all RFP requirements, and is compelling to the grant committee.",
            allow_delegation=True,
            verbose=True,
            llm=self.get_groq_llm(),
        )
