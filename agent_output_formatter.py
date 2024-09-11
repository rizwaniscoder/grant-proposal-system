# agent_output_formatter.py

import textwrap

def format_agent_output(data):
    output = []

    # Header
    output.append("📋 RFP / Proposal Draft")
    output.append("---")

    # Background
    if 'background' in data:
        output.append("🎯 Background:")
        output.append(textwrap.fill(data['background'], width=80))
        output.append("")

    # Grant Questions
    if 'questions' in data:
        output.append("📝 Grant Questions:")
        for i, question in enumerate(data['questions'], 1):
            wrapped_question = textwrap.fill(question, width=76, subsequent_indent='   ')
            output.append(f"{i}. {wrapped_question}")
        output.append("")

    # Uploaded Files
    if 'uploaded_files' in data:
        output.append("📎 Uploaded Files:")
        for file in data['uploaded_files']:
            output.append(f"   • {file['name']} ({file['size']})")
        output.append("")

    # File Upload Instructions
    output.append("📤 File Upload Instructions:")
    output.append("   • Drag and drop files here")
    output.append("   • Limit 200MB per file • PDF format")
    output.append("")

    # Status
    if 'status' in data:
        output.append(f"🔄 Status: {data['status']}")

    return "\n".join(output)

def simulate_agent_task(input_data):
    # This function simulates an agent processing some input data
    # In a real scenario, this would be where your agent does its actual work
    output_data = {
        'background': input_data.get('background', ''),
        'questions': input_data.get('questions', []),
        'uploaded_files': input_data.get('uploaded_files', []),
        'status': "Processing complete"
    }
    return output_data

def run_agent_with_formatting(input_data):
    # Process the input with the agent
    agent_output = simulate_agent_task(input_data)
    
    # Format the output
    formatted_output = format_agent_output(agent_output)
    
    # In a real application, you might send this to a UI
    # For this example, we'll just print it
    print(formatted_output)

# Example usage
if __name__ == "__main__":
    sample_input = {
        'background': "Please provide background on the RFP / proposal that needs to be drafted for the new community center project.",
        'questions': [
            "Tell us more about your organization and its mission. (150 words)",
            "How would you describe the community center project in a sentence? (50 words)",
            "What is the total project budget for the community center?",
            "For what specific aspects of the community center project are you seeking support?",
            "Describe the activities that would be carried out with support from the grant. (250 Words)",
            "What outcomes do you expect from successfully implementing these activities? (250 Words)",
            "How will you measure and evaluate the success of this project?"
        ],
        'uploaded_files': [
            {'name': "Community_Center_Proposal_Draft.pdf", 'size': "1.2MB"},
            {'name': "Budget_Breakdown.xlsx", 'size': "156KB"}
        ]
    }

    run_agent_with_formatting(sample_input)
