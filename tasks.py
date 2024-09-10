from crewai import Task
from textwrap import dedent

class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def document_ingestion_task(self, agent, var1, var2):
        return Task(
            description=dedent(
                f"""
                Process and organize uploaded files.
                {self.__tip_section()}
                Use these variables: 
                Organization name: {var1}
                RFP / Proposal background: {var2}
                """
            ),
            agent=agent,
            expected_output="Organized document files and metadata",
        )

    def rfp_analysis_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Analyze RFP requirements and create detailed task outlines.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Detailed task outlines based on RFP requirements",
        )

    def mission_vision_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Craft compelling narratives around the nonprofit's mission, vision, and impact.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Mission and vision statements",
            output_file="mission_vision.md",
        )

    def impact_research_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Gather and analyze impact data.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Impact data analysis and summary",
            output_file="impact_research.md",
        )

    def budget_analysis_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Review financial documents and create budget narratives.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Budget analysis and narratives",
            output_file="budget_analysis.md",
        )

    def team_governance_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Highlight the nonprofit's team and governance strengths.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Team and governance summary",
            output_file="team_governance.md",
        )

    def case_testimonial_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Create compelling case statements and testimonials.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Case statements and testimonials",
            output_file="case_testimonial.md",
        )

    def quality_integration_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Ensure consistency and quality across all outputs.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Quality assurance reports",
            output_file="quality_integration.md",
        )

    def formatting_submission_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Organize the final document based on agent outputs.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Final formatted document",
            output_file="formatting_submission.md",
        )

    def project_manager_task(self, agent):
        return Task(
            description=dedent(
                f"""
                Organize tasks and oversee the creation of the full grant proposal.
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Project management plan and updates",
        )
