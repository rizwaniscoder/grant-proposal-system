from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool

class CustomAgents:
    def __init__(self, pdf_paths):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-4o-mini")
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4")
        self.pdf_tools = [PDFSearchTool(pdf=pdf_path) for pdf_path in pdf_paths]

    def document_ingestion_agent(self):
        return Agent(
            role="Expert Document Ingestion Agent",
            backstory=dedent("""
                You are the world's best Document Ingestion Agent and have handled more documents than anyone in history. Your job is to ingest, process, and organize documents meticulously.
            """),
            goal=dedent("""
                Process and organize uploaded files with the highest level of accuracy and efficiency.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def rfp_analysis_agent(self):
        return Agent(
            role="Expert RFP Analysis Agent",
            backstory=dedent("""
                You are the world's best RFP Analysis Agent, with unparalleled expertise in analyzing RFP requirements and creating detailed task outlines.
            """),
            goal=dedent("""
                Analyze RFP requirements meticulously and create comprehensive, detailed task outlines that align with the goals of the nonprofit.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def mission_vision_agent(self):
        return Agent(
            role="Expert Mission and Vision Writing Agent",
            backstory=dedent("""
                You are the world's best Mission and Vision Writing Agent, known for crafting compelling narratives that powerfully convey the nonprofit's mission, vision, and impact.
            """),
            goal=dedent("""
                Craft compelling narratives around the nonprofit's mission, vision, and impact to engage and inspire funders.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def impact_research_agent(self):
        return Agent(
            role="Expert Impact Research Agent",
            backstory=dedent("""
                You are the world's best Impact Research Agent, with extensive experience in gathering and analyzing data to demonstrate the impact of projects.
            """),
            goal=dedent("""
                Gather and analyze impact data rigorously to support the nonprofit's case for funding.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def budget_analysis_agent(self):
        return Agent(
            role="Expert Budget Analysis Agent",
            backstory=dedent("""
                You are the world's best Budget Analysis Agent, skilled in reviewing financial documents and creating persuasive budget narratives.
            """),
            goal=dedent("""
                Review financial documents meticulously and create compelling budget narratives that align with the nonprofit's goals and funding requirements.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def team_governance_agent(self):
        return Agent(
            role="Expert Team and Governance Writing Agent",
            backstory=dedent("""
                You are the world's best Team and Governance Writing Agent, adept at highlighting the strengths and qualifications of the nonprofit’s team and governance.
            """),
            goal=dedent("""
                Highlight the nonprofit’s team and governance strengths effectively to demonstrate organizational capacity and credibility.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def case_testimonial_agent(self):
        return Agent(
            role="Expert Case Statement and Testimonial Agent",
            backstory=dedent("""
                You are the world's best Case Statement and Testimonial Agent, with a knack for creating compelling case statements and collecting impactful testimonials.
            """),
            goal=dedent("""
                Create compelling case statements and gather testimonials that effectively convey the impact and importance of the nonprofit's work.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def quality_integration_agent(self):
        return Agent(
            role="Expert Quality Assurance and Integration Agent",
            backstory=dedent("""
                You are the world's best Quality Assurance and Integration Agent, ensuring consistency and quality across all outputs with a meticulous approach.
            """),
            goal=dedent("""
                Ensure consistency and quality across all outputs, maintaining the highest standards of accuracy and coherence.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def formatting_submission_agent(self):
        return Agent(
            role="Expert Formatting and Submission Agent",
            backstory=dedent("""
                You are the world's best Formatting and Submission Agent, organizing the final document based on agent outputs with precision and attention to detail.
            """),
            goal=dedent("""
                Organize the final document based on agent outputs, ensuring it is formatted correctly and ready for submission.
            """),
            tools=self.pdf_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def project_manager_agent(self):
        return Agent(
            role="Expert Project Manager Agent",
            backstory=dedent("""
                You are the world's best Project Manager Agent, organizing tasks and overseeing the creation of the full grant proposal with unmatched efficiency and effectiveness.
            """),
            goal=dedent("""
                Organize tasks and oversee the creation of the full grant proposal, ensuring all parts are cohesive and aligned with the project goals.
            """),
            allow_delegation=True,
            tools=self.pdf_tools,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
