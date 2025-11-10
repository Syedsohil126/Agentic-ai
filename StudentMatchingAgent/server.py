import os
import csv
from flask import Flask, render_template_string, jsonify
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# The LLM setup is now a global variable to be used by the app
# You MUST replace the placeholder below with your actual API key.
llm = LLM(
    api_key="AIzaSyCh8fTvoaaQiJGk213EKHgA2C2Ckbgx-74",
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
                # Use .strip() to handle any potential whitespace in column names
                students.append(
                    f"Student {row['name'].split()[-1]}: Course = {row['course'].strip()}, Academic Performance = {row['academic_performance'].strip()}, Schedule = {row['schedule'].strip()}"
                )
    except FileNotFoundError:
        return "Student data file not found."
    return "\n".join(students)

def run_matching_agent():
    """Runs the CrewAI agent to match students."""
    # The file path is now correctly pointing to the 'src' directory
    student_data = read_student_data('src/students.csv')
    
    # Check if the student data is blank or if the file was not found
    if not student_data or "not found" in student_data:
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
        # Return the actual error message
        print(f"An unhandled error occurred: {e}")
        return f"An error occurred during crew execution: {str(e)}"

# The main web page route
@app.route('/')
def home():
    """Serves the main web page with a button to run the agent."""
    react_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Student Matching Agent</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4">
            <div id="root" class="max-w-4xl w-full"></div>

            <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
            <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
            <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

            {% raw %}
            <script type="text/babel">
                const { useState } = React;

                const App = () => {
                    const [results, setResults] = useState('');
                    const [isLoading, setIsLoading] = useState(false);
                    const [error, setError] = useState(null);

                    const runAgent = async () => {
                        setIsLoading(true);
                        setError(null);
                        setResults('');
                        try {
                            const response = await fetch('/run', { method: 'POST' });
                            const data = await response.json();
                            if (response.ok) {
                                setResults(data.output);
                            } else {
                                setError(data.output || 'An unknown error occurred.');
                            }
                        } catch (err) {
                            setError('An error occurred. Please check the server logs.');
                            console.error('Fetch error:', err);
                        } finally {
                            setIsLoading(false);
                        }
                    };

                    return (
                        <div className="w-full bg-white p-8 rounded-lg shadow-lg">
                            <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">Student Matching Agent</h1>
                            <p className="text-gray-600 mb-8 text-center">
                                Click the button below to run the AI agent and get student pairing recommendations.
                            </p>
                            <div className="flex justify-center mb-8">
                                <button
                                    onClick={runAgent}
                                    disabled={isLoading}
                                    className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out disabled:bg-gray-400"
                                >
                                    {isLoading ? 'Running...' : 'Run Matching Agent'}
                                </button>
                            </div>
                            <div className="mt-8">
                                <h2 className="text-xl font-semibold mb-4 text-gray-800" style={{ display: results || error ? 'block' : 'none' }}>Recommendations:</h2>
                                <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 whitespace-pre-wrap">
                                    {isLoading && <p className="text-gray-500 italic">Analyzing student data and generating recommendations...</p>}
                                    {error && <p className="text-red-500">{error}</p>}
                                    {results && <p>{results}</p>}
                                </div>
                            </div>
                        </div>
                    );
                };

                const root = ReactDOM.createRoot(document.getElementById('root'));
                root.render(<App />);
            </script>
            {% endraw %}
        </body>
        </html>
    """
    return render_template_string(react_html)

# The API endpoint to run the agent
@app.route('/run', methods=['POST'])
def run_agent_endpoint():
    """An API endpoint to trigger the agent and return the result as JSON."""
    result = run_matching_agent()
    # The 'output' key must be in the JSON response for the React app to work.
    if "An error occurred" in result:
        # Return the error message with a 400 status code
        return jsonify({"output": result}), 400
    if "not found" in result:
        return jsonify({"output": result}), 400
    return jsonify({"output": str(result)})

if __name__ == '__main__':
    # You may need to install flask with: pip install Flask
    app.run(debug=True, port=8000)
