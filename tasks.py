from crewai import Agent, Task, Crew
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

    def mission_vision_task(self, agent):
        return Task(
            description=dedent(f"""
                Craft compelling narratives around the nonprofit's mission, vision, and impact.
                
                Your task is to:
                1. Summarize the organization's mission and vision based on provided information.
                2. Highlight key aspects of the organization's impact and achievements.
                3. Create a narrative that aligns the organization's goals with the grant's objectives.
                4. Ensure the language is inspiring and compelling for potential funders.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Compelling mission and vision statements with impact narrative",
        )

    def impact_research_task(self, agent):
        return Task(
            description=dedent(f"""
                Gather and analyze impact data to support the nonprofit's case for funding.
                
                Your task is to:
                1. Identify key impact metrics relevant to the organization's work.
                2. Collect data on these metrics from provided documents and any additional research.
                3. Analyze the data to demonstrate the organization's effectiveness.
                4. Present the findings in a clear, compelling format suitable for the grant proposal.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive impact analysis with supporting data",
        )

    def budget_analysis_task(self, agent):
        return Task(
            description=dedent(f"""
                Review financial documents and create a persuasive budget narrative.
                
                Your task is to:
                1. Analyze the organization's financial documents.
                2. Create a detailed budget for the proposed project or general operations.
                3. Write a narrative explaining and justifying each budget item.
                4. Ensure the budget aligns with the grant requirements and organization's goals.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Detailed budget with persuasive narrative explanation",
        )

    def team_governance_task(self, agent):
        return Task(
            description=dedent(f"""
                Highlight the nonprofit's team and governance strengths.
                
                Your task is to:
                1. Identify key team members and their qualifications.
                2. Describe the organization's governance structure.
                3. Highlight any unique strengths or experiences of the team and board.
                4. Explain how the team and governance contribute to the organization's effectiveness.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive overview of team strengths and governance structure",
        )

    def case_testimonial_task(self, agent):
        return Task(
            description=dedent(f"""
                Create compelling case statements and gather impactful testimonials.
                
                Your task is to:
                1. Develop a strong case statement for why the organization deserves funding.
                2. Identify potential sources for testimonials (e.g., beneficiaries, partners).
                3. Draft sample testimonials or quotes that support the organization's impact.
                4. Integrate case statements and testimonials into a cohesive narrative.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Persuasive case statements with supporting testimonials",
        )

    def quality_integration_task(self, agent):
        return Task(
            description=dedent(f"""
                Ensure consistency and quality across all outputs.
                
                Your task is to:
                1. Review all sections of the grant proposal for consistency in tone and messaging.
                2. Check for any contradictions or redundancies across sections.
                3. Ensure all required elements of the grant application are addressed.
                4. Improve the overall flow and readability of the document.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Quality assurance report with suggested improvements",
        )

    def formatting_submission_task(self, agent):
        return Task(
            description=dedent(f"""
                Organize and format the final document for submission.
                
                Your task is to:
                1. Compile all sections of the grant proposal into a single document.
                2. Format the document according to any specified guidelines.
                3. Create a table of contents and ensure proper page numbering.
                4. Proofread the entire document for any final errors or improvements.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Final formatted and proofread grant proposal ready for submission",
        )

    def project_manager_task(self, agent):
        return Task(
            description=dedent(f"""
                Oversee the entire grant proposal creation process.
                
                Your task is to:
                1. Coordinate the efforts of all other agents and tasks.
                2. Ensure all deadlines are met and tasks are completed in the correct order.
                3. Resolve any conflicts or issues that arise during the process.
                4. Provide regular updates on the progress of the grant proposal.
                
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Project management plan and final report on grant proposal creation process",
        )
