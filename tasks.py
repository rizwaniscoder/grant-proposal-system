from crewai import Task
from textwrap import dedent

class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def document_ingestion_task(self, agent, org_name, proposal_background):
        return Task(
            description=dedent(f"""
                Process and organize uploaded files for {org_name}.
                Background information on the RFP/proposal:
                {proposal_background}
                
                Your task is to thoroughly read and analyze all uploaded documents,
                extracting key information relevant to the grant proposal.
                Organize this information in a clear, structured format.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Organized summary of key information from all uploaded documents",
        )

    def rfp_analysis_task(self, agent):
        return Task(
            description=dedent(f"""
                Analyze the RFP (Request for Proposal) requirements and create detailed task outlines.
                
                Your task is to:
                1. Identify all key requirements stated in the RFP.
                2. Break down these requirements into specific, actionable tasks.
                3. Create a comprehensive outline of tasks needed to complete the grant proposal.
                4. Prioritize these tasks based on importance and deadline (if provided).
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Detailed task outline based on RFP requirements",
        )

    def proposal_writing_task(self, agent, rfp_analysis, org_name):
        return Task(
            description=dedent(f"""
                Write a comprehensive grant proposal for {org_name} based on the RFP analysis.
                RFP Analysis: {rfp_analysis}
                
                Your task is to:
                1. Use the RFP analysis to structure the proposal.
                2. Address all key requirements identified in the RFP.
                3. Highlight the organization's strengths and alignment with the grant's objectives.
                4. Ensure the proposal is well-written, persuasive, and tailored to the specific grant.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive grant proposal draft",
        )

    def budget_preparation_task(self, agent, proposal_content, total_budget):
        return Task(
            description=dedent(f"""
                Based on the proposal content and the total budget of ${total_budget:,}, create an itemized budget. 
                Ensure that the sum of all items equals the total budget. 
                Consider the specific needs and requirements mentioned in the proposal.
                Proposal content: {proposal_content}
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Detailed itemized budget breakdown based on the proposal content and total budget",
        )

    def quality_review_task(self, agent, proposal_draft, budget):
        return Task(
            description=dedent(f"""
                Review the complete grant proposal draft and budget for quality assurance.
                Proposal Draft: {proposal_draft}
                Budget: {budget}
                
                Your task is to:
                1. Ensure all RFP requirements are adequately addressed.
                2. Check for clarity, coherence, and persuasiveness of the proposal.
                3. Verify that the budget aligns with the proposal content and grant requirements.
                4. Suggest any final improvements or adjustments.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Quality assurance report with suggestions for improvements",
        )