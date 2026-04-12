import os
import nest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Setup Environment
nest_asyncio.apply()
load_dotenv()

# 2. Setup FastAPI
app = FastAPI(title="Retail Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Native CrewAI LLM & Tools
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GOOGLE_API_KEY")
)

@tool("Web Search")
def search_tool(query: str) -> str:
    """Searches the live internet for information."""
    return DuckDuckGoSearchRun().run(query)

# 4. Define the Expected User Input
class ResearchQuery(BaseModel):
    topic: str

# 5. The API Endpoint
@app.post("/research")
async def run_research(query: ResearchQuery):
    print(f"\n--- API Request Received! Researching: {query.topic} ---")
    
    # Define Agents
    researcher = Agent(
        role="Senior Retail Market Researcher",
        goal="Discover the most current and impactful technology trends in the retail industry.",
        backstory="You are an expert market analyst. You know exactly how to search the live internet to find emerging trends, customer behaviors, and tech advancements in retail.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=gemini_llm
    )

    analyst = Agent(
        role="Lead Retail Analyst & Writer",
        goal="Synthesize raw internet research into a clear, professional, and actionable business report.",
        backstory="You are a meticulous writer. You take messy research notes and turn them into beautifully formatted executive summaries for the CEO.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # Define Tasks using the User's Topic!
    research_task = Task(
        description=f"Search the internet to comprehensively research the following topic in the retail industry: '{query.topic}'. Find specific, real-world examples and high-quality sources.",
        expected_output="A detailed bulleted list of findings, trends, and real-world examples related to the topic.",
        agent=researcher
    )

    write_task = Task(
        description=f"Take the findings from the Senior Researcher about '{query.topic}' and write a formal executive summary. Format it nicely with headers.",
        expected_output="A professionally formatted text report ready for business executives.",
        agent=analyst,
        output_file="retail_report.txt"
    )

    # Build and Run Crew
    retail_crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, write_task],
        verbose=True,
        process=Process.sequential
    )

    # Kickoff the agents (This takes 20-40 seconds to finish)
    retail_crew.kickoff()

    # Read the saved file and send it back to the HTML frontend
    with open("retail_report.txt", "r", encoding="utf-8") as file:
        final_report = file.read()

    return {"report": final_report}