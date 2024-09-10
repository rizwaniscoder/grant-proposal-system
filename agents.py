from crewai import Agent
from textwrap import dedent
from langchain_groq import ChatGroq
import os
from langchain.tools import Tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class CustomAgents:
    def __init__(self, pdf_paths):
        self.groq_llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.pdf_tools = [self.create_pdf_search_tool(pdf_path) for pdf_path in pdf_paths]

    def create_pdf_search_tool(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        return Tool(
            name=f"Search {pdf_path}",
            func=lambda q: db.similarity_search(q, k=1)[0].page_content,
            description=f"Useful for searching content in {pdf_path}"
        )

    def document_ingestion_agent(self):
        return Agent(
            role="Expert Document Ingestion Agent",
            backstory="You are the world's best Document Ingestion Agent and have handled more documents than anyone in history. Your job is to ingest, process, and organize documents meticulously.",
            goal="Process and organize uploaded files with the highest level of accuracy and efficiency.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def rfp_analysis_agent(self):
        return Agent(
            role="Expert RFP Analysis Agent",
            backstory="You are the world's best RFP Analysis Agent, with unparalleled expertise in analyzing RFP requirements and creating detailed task outlines.",
            goal="Analyze RFP requirements meticulously and create comprehensive, detailed task outlines that align with the goals of the nonprofit.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def mission_vision_agent(self):
        return Agent(
            role="Expert Mission and Vision Writing Agent",
            backstory="You are the world's best Mission and Vision Writing Agent, known for crafting compelling narratives that powerfully convey the nonprofit's mission, vision, and impact.",
            goal="Craft compelling narratives around the nonprofit's mission, vision, and impact to engage and inspire funders.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def impact_research_agent(self):
        return Agent(
            role="Expert Impact Research Agent",
            backstory="You are the world's best Impact Research Agent, with extensive experience in gathering and analyzing data to demonstrate the impact of projects.",
            goal="Gather and analyze impact data rigorously to support the nonprofit's case for funding.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def budget_analysis_agent(self):
        return Agent(
            role="Expert Budget Analysis Agent",
            backstory="You are the world's best Budget Analysis Agent, skilled in reviewing financial documents and creating persuasive budget narratives.",
            goal="Review financial documents meticulously and create compelling budget narratives that align with the nonprofit's goals and funding requirements.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def team_governance_agent(self):
        return Agent(
            role="Expert Team and Governance Writing Agent",
            backstory="You are the world's best Team and Governance Writing Agent, adept at highlighting the strengths and qualifications of the nonprofit's team and governance.",
            goal="Highlight the nonprofit's team and governance strengths effectively to demonstrate organizational capacity and credibility.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def case_testimonial_agent(self):
        return Agent(
            role="Expert Case Statement and Testimonial Agent",
            backstory="You are the world's best Case Statement and Testimonial Agent, with a knack for creating compelling case statements and collecting impactful testimonials.",
            goal="Create compelling case statements and gather testimonials that effectively convey the impact and importance of the nonprofit's work.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def quality_integration_agent(self):
        return Agent(
            role="Expert Quality Assurance and Integration Agent",
            backstory="You are the world's best Quality Assurance and Integration Agent, ensuring consistency and quality across all outputs with a meticulous approach.",
            goal="Ensure consistency and quality across all outputs, maintaining the highest standards of accuracy and coherence.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def formatting_submission_agent(self):
        return Agent(
            role="Expert Formatting and Submission Agent",
            backstory="You are the world's best Formatting and Submission Agent, organizing the final document based on agent outputs with precision and attention to detail.",
            goal="Organize the final document based on agent outputs, ensuring it is formatted correctly and ready for submission.",
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.groq_llm,
        )

    def project_manager_agent(self):
        return Agent(
            role="Expert Project Manager Agent",
            backstory="You are the world's best Project Manager Agent, organizing tasks and overseeing the creation of the full grant proposal with unmatched efficiency and effectiveness.",
            goal="Organize tasks and oversee the creation of the full grant proposal, ensuring all parts are cohesive and aligned with the project goals.",
            allow_delegation=True,
            tools=self.pdf_tools,
            verbose=True,
            llm=self.groq_llm,
        )
