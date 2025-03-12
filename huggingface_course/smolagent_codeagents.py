from smolagents import (
    CodeAgent,
    tool,
    LiteLLMModel,
    DuckDuckGoSearchTool,
)
import sys


@tool
def calculate(expression: str) -> int:
    """
    Calculates the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to evaluate.
    """
    return eval(expression)


model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="ollama",
)

search_tool = DuckDuckGoSearchTool(max_results=5)

agent = CodeAgent(
    model=model,
    tools=[search_tool, calculate],
    additional_authorized_imports=["datetime", "numpy", "pandas"],
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent.run(sys.argv[1])
    else:
        print("Please provide an argument")

# Example CLI
# python smolagent_codeagents.py "What is the capital of France?"
# python smolagent_codeagents.py "Calculate 1 + two multiply 3"
# python smolagent_codeagents.py "What time is it now?"
