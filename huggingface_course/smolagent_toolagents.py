from smolagents import (
    load_tool,
    Tool,
    tool,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    ToolCallingAgent,
)
import sys
import subprocess


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

# image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt",
)

agent = ToolCallingAgent(
    model=model, tools=[search_tool, calculate, image_generation_tool]
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent.run(sys.argv[1])
    else:
        print("Please provide an argument")

# Example CLI
# python simple_toolagents.py "What is the capital of France?"
# python simple_toolagents.py "Calculate 1 + two multiply 3"
# python simple_toolagents.py "What time is it now?"
