import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file
load_dotenv()

class StudentMatchingCrew:
    def __init__(self):
        # Use the LLM wrapper class from CrewAI
        self.llm = LLM(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini/gemini-2.5-flash",
            temperature=0.1
        )
    
    def run(self):
        data = """
        Student A: Course = CS101, Academic Performance = Good, Schedule = Mon/Wed evenings
        Student B: Course = CS101, Academic Performance = Excellent, Schedule = Tue/Thu afternoons
        Student C: Course = CS102, Academic Performance = Average, Schedule = Mon/Wed evenings
        Student D: Course = CS101, Academic Performance = Good, Schedule = Mon/Wed evenings
        """
        
        matcher_agent = Agent(
            role='Student Matcher',
            goal='Identify and suggest optimal student pairings for collaboration based on academic needs and schedules.',
            backstory='An expert in educational technology and collaborative learning. Your job is to analyze student data and find the best matches for study groups, project collaboration, or peer tutoring.',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        matching_task = Task(
            description=f"Analyze the following student data and provide a list of recommended matches. The recommendations should be justified based on course, academic performance, and schedule. Specifically, find peers for study groups (same course, similar schedule), project collaboration (similar skills, complementary schedules), or peer tutoring (one student needs help, another is strong in that area).\n\nStudent Data:\n{data}",
            agent=matcher_agent,
            expected_output='A clear, bulleted list of recommended student pairings with a brief explanation for each pairing. Example: "- Student A and Student D: Both are in CS101 and have matching schedules, making them perfect for a study group."'
        )

        crew = Crew(
            agents=[matcher_agent],
            tasks=[matching_task],
            process=Process.sequential,
            verbose=True,
            full_output=True, # Note: This parameter is for detailed logs, not the final result object structure
        )

        result = crew.kickoff()
        print("\n\n################################")
        print("Final Matching Recommendations:")
        print("################################\n")
        print(result) # Corrected from print(result['final_output'])

if __name__ == "__main__":
    print("## Student Collaboration Agent ##")
    StudentMatchingCrew().run()