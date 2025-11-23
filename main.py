from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()  # Load environment variables from .env file

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# LLM
llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)

# Parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that provides concise summaries on given topics."),
    ("user", 
        "Provide a summary on the following topic along with sources and tools used: {topic}\n"
        "Respond in the following JSON format:\n"
        "{format_instructions}"
    )
])

# Build chain using LCEL
chain = (
    prompt
    | llm
    | parser
)

# Run the chain
response = chain.invoke({
    "topic": "The impact of climate change on global agriculture",
    "format_instructions": parser.get_format_instructions()
})

print(response)
