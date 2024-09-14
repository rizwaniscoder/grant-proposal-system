# RFP / Proposal Draft Generator

## 📋 Project Overview

This Streamlit application uses AI-powered agents to analyze Request for Proposal (RFP) documents and generate proposal drafts. It leverages the CrewAI framework and Groq's large language model to process uploaded PDFs, extract key information, and produce organized summaries and proposal outlines.

## 🚀 Features

- Upload and process multiple PDF documents
- AI-powered analysis of RFP requirements
- Real-time display of AI agent activities and thoughts
- Generation of organized summaries and proposal drafts
- Download capability for analysis results
- Input total budget and generate itemized budget breakdown
- Real-time display of AI agent activities, thoughts, and outputs

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rfp-proposal-generator.git
   cd rfp-proposal-generator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## 🚀 Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter the organization name and provide background information on the RFP/proposal.

4. Upload the relevant PDF documents (max 200MB each).

5. Click the "Run Custom Crew" button to start the analysis.

6. Watch the real-time output as the AI agents process the documents and generate the proposal draft.

7. Once complete, you can view the analysis result and download it as a JSON file.

## 🧩 Project Structure

- `streamlit_app.py`: Main Streamlit application file
- `agents.py`: Defines the AI agents used in the analysis
- `tasks.py`: Defines the tasks performed by the agents
- `requirements.txt`: List of Python dependencies

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/rfp-proposal-generator/issues).

## 📝 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Groq](https://groq.com/)
- [LangChain](https://langchain.com/)
