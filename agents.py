from crewai import Agent
from textwrap import dedent
from langchain_groq import ChatGroq
import os
from langchain.tools import Tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class CustomAgents:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.groq_llm = None
        self.pdf_tools = {}
        self.embeddings = None

    def get_groq_llm(self):
        if self.groq_llm is None:
            self.groq_llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                api_key=os.getenv("GROQ_API_KEY")
            )
        return self.groq_llm

    def get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return self.embeddings

    def create_pdf_search_tool(self, pdf_path):
        if pdf_path not in self.pdf_tools:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            db = FAISS.from_documents(texts, self.get_embeddings())

            self.pdf_tools[pdf_path] = Tool(
                name=f"Search {pdf_path}",
                func=lambda q, db=db: db.similarity_search(q, k=1)[0].page_content,
                description=f"Useful for searching content in {pdf_path}"
            )
        return self.pdf_tools[pdf_path]

    def get_pdf_tools(self):
        return [self.create_pdf_search_tool(path) for path in self.pdf_paths]

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

    # Other agent methods can be added back as needed, following the same pattern
