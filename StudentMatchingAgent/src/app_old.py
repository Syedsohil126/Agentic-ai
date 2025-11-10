import os
import csv
from flask import Flask, render_template_string
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# The LLM setup is now a global variable to be used by the app
llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-2.5-flash",
    temperature=0.1
)

def read_student_data(file_path):
    """Reads student data from a CSV file and returns it as a formatted string."""
    students = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append(
                    f"Student {row['name'].split()[-1]}: Course = {row['course']}, Academic Performance = {row['academic_performance']}, Schedule = {row['schedule']}"
                )
    except FileNotFoundError:
        return "Student data file not found."
    return "\n".join(students)

def run_matching_agent():
    """Runs the CrewAI agent to match students."""
    student_data = read_student_data('students.csv')
    
    # Check if the student data is blank or if the file was not found
    if not student_data:
        return "Student data file is blank or data could not be read. Please add student information to students.csv."
    
    if "not found" in student_data:
        return student_data
    
    matcher_agent = Agent(
        role='Student Matcher',
        goal='Identify and suggest optimal student pairings for collaboration based on academic needs and schedules.',
        backstory='An expert in educational technology and collaborative learning. Your job is to analyze student data and find the best matches for study groups, project collaboration, or peer tutoring.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    matching_task = Task(
        description=f"Analyze the following student data and provide a list of recommended matches. The recommendations should be justified based on course, academic performance, and schedule. Specifically, find peers for study groups (same course, similar schedule), project collaboration (similar skills, complementary schedules), or peer tutoring (one student needs help, another is strong in that area).\n\nStudent Data:\n{student_data}",
        agent=matcher_agent,
        expected_output='A clear, bulleted list of recommended student pairings with a brief explanation for each pairing. Example: "- Student A and Student D: Both are in CS101 and have matching schedules, making them perfect for a study group."'
    )

    crew = Crew(
        agents=[matcher_agent],
        tasks=[matching_task],
        process=Process.sequential,
        verbose=True,
        full_output=True,
    )
    
    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        return f"An error occurred during crew execution: {str(e)}"

# The main web page route
@app.route('/')
def home():
    """Serves the main web page with a button to run the agent."""
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Student Matching Agent</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4">
            <div class="max-w-4xl w-full bg-white p-8 rounded-lg shadow-lg">
                <h1 class="text-3xl font-bold mb-6 text-center text-gray-800">Student Matching Agent</h1>
                <p class="text-gray-600 mb-8 text-center">
                    Click the button below to run the AI agent and get student pairing recommendations.
                </p>
                <div class="flex justify-center mb-8">
                    <button id="run-button" onclick="runAgent()" class="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out">
                        Run Matching Agent
                    </button>
                </div>
                <div id="results-container" class="mt-8">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800 hidden" id="results-title">Recommendations:</h2>
                    <div id="results-content" class="bg-gray-50 p-6 rounded-lg border border-gray-200">
                        <p id="placeholder" class="text-gray-500 italic">Results will appear here...</p>
                    </div>
                </div>
            </div>

            <script>
                async function runAgent() {
                    const runButton = document.getElementById('run-button');
                    const resultsContent = document.getElementById('results-content');
                    const placeholder = document.getElementById('placeholder');
                    const resultsTitle = document.getElementById('results-title');

                    runButton.disabled = true;
                    runButton.innerText = 'Running...';
                    placeholder.innerText = 'Analyzing student data and generating recommendations...';
                    placeholder.classList.remove('italic', 'text-gray-500');
                    resultsTitle.classList.add('hidden');

                    try {
                        const response = await fetch('/run', { method: 'POST' });
                        const data = await response.json();
                        
                        resultsContent.innerText = data.output;
                        resultsTitle.classList.remove('hidden');

                    } catch (error) {
                        resultsContent.innerText = 'An error occurred. Please check the server logs.';
                        console.error('Error:', error);
                    } finally {
                        runButton.disabled = false;
                        runButton.innerText = 'Run Matching Agent';
                    }
                }
            </script>
        </body>
        </html>
    ''')

# The API endpoint to run the agent
@app.route('/run', methods=['POST'])
def run_agent_endpoint():
    """An API endpoint to trigger the agent and return the result as JSON."""
    result = run_matching_agent()
    return {"output": result}

if __name__ == '__main__':
    # You may need to install flask with: pip install Flask
    app.run(debug=True)
