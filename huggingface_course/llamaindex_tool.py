from llama_index.tools.google import GmailToolSpec
from rich.console import Console
from rich.table import Table
from llama_index.llms.ollama import Ollama

console = Console()


def print_tool_list():
    table = Table(title="Gmail Tools")
    table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    for tool in tool_spec_list:
        table.add_row(tool.metadata.name, tool.metadata.description)

    console.print(table)


tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()
print_tool_list()


llm = Ollama(model="qwen2.5:7b", request_timeout=1200)

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.agent import ReActAgent
import gradio as gr

agent = ReActAgent.from_tools(tools=tool_spec_list, llm=llm, verbose=True)


def chat_with_agent(message, history):
    try:
        response = agent.chat(message)
        return str(response)
    except Exception as e:
        print(e)
        return "Error, please try again"


demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Gmail Assistant",
    description="Chat with an AI assistant that can help you with Gmail tasks",
)

if __name__ == "__main__":
    demo.launch()
