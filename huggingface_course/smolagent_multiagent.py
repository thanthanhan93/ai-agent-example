from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, ToolCallingAgent

model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="ollama",
)

search_tool = DuckDuckGoSearchTool(max_results=5)

web_agent = ToolCallingAgent(
    model=model,
    tools=[search_tool],
    name="web_agent",
    description="Browses the web to find information",
    max_steps=2,
)

agent = CodeAgent(
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["curl"],
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent.run(sys.argv[1])
    else:
        print("Please provide an argument")
