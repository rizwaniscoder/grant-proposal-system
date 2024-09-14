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

    def proposal_writing_task(self, agent, rfp_analysis, org_info):
        return Task(
            description=dedent(f"""
                Write a compelling proposal based on the RFP analysis and organization information.
                
                Your task is to:
                1. Use the RFP analysis to address all key requirements.
                2. Incorporate the organization's background, goals, and unique value proposition.
                3. Structure the proposal logically and persuasively.
                4. Ensure the language is clear, concise, and tailored to the grant committee.
                5. Include relevant statistics, case studies, or success stories to support your points.
                
                RFP Analysis: {rfp_analysis}
                Organization Info: {org_info}
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Complete draft of the grant proposal, addressing all RFP requirements and highlighting the organization's strengths",
        )

    def budget_preparation_task(self, agent, proposal_draft):
        return Task(
            description=dedent(f"""
                Prepare a detailed budget for the grant proposal.
                
                Your task is to:
                1. Review the proposal draft and identify all budget-related items.
                2. Create a comprehensive budget that covers all aspects of the proposed project.
                3. Ensure the budget is realistic, justified, and aligns with the RFP requirements.
                4. Provide brief explanations for each budget item.
                5. Include any required matching funds or in-kind contributions.
                6. Ensure the budget adheres to any specific formatting or category requirements in the RFP.
                
                Proposal Draft: {proposal_draft}
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Detailed budget spreadsheet with itemized costs, justifications, and any required explanatory notes",
        )

    def quality_review_task(self, agent, full_proposal):
        return Task(
            description=dedent(f"""
                Review and improve the complete grant proposal.
                
                Your task is to:
                1. Ensure all RFP requirements are met and clearly addressed.
                2. Check for consistency in tone, style, and messaging throughout the proposal.
                3. Verify that the budget aligns with the proposal narrative.
                4. Suggest improvements for clarity, persuasiveness, and overall quality.
                5. Proofread for any grammatical or formatting errors.
                6. Ensure all required attachments or supplementary materials are included and properly referenced.
                7. Verify that the proposal adheres to any page limits or formatting requirements specified in the RFP.
                
                Full Proposal: {full_proposal}
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive review report with specific suggestions for improvements, corrections, and final polish of the grant proposal",
        )