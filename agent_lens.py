from openai import OpenAI
from duckduckgo_search import DDGS
import json, re

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

query = input("Describe your agentic workflow: ")

# Step 1: Web search using DuckDuckGo
print("\n Searching the web...\n")
results = DDGS().text(f"best LLM models for {query} 2025", max_results=5)
search_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])

# Step 2: Send query + search results to the local model
response = client.responses.create(
    model="minimax-m2:cloud",
    instructions="""You are an LLM recommendation expert.
    You will be given web search results and a user query.
    Based on the search results, return ONLY a JSON array of 5 LLMs best suited for the workflow.
    Each object must have: llm_name, provider, parameters, tool_calling_support, suitability_score (1-10).
    No explanation, no markdown, just the raw JSON array.""",
    input=f"User Query: {query}\n\nWeb Search Results:\n{search_context}",
)

raw = re.sub(r"```(?:json)?", "", response.output_text).strip().strip("`").strip()
result = json.loads(raw)

for i, llm in enumerate(result, 1):
    print(f"#{i} {llm['llm_name']} ({llm['provider']})")
    print(f"   Parameters       : {llm['parameters']}")
    print(f"   Tool Calling     : {llm['tool_calling_support']}")
    print(f"   Suitability Score: {llm['suitability_score']}/10\n")  