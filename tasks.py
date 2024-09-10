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

    # Other task methods can be added back as needed, following the same pattern
