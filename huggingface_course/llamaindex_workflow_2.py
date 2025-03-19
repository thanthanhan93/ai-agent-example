from typing import Any, Dict, Optional
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.workflow import Context
import asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow

from rich.logging import RichHandler
from rich.console import Console
import logging
import sys
import llama_index.core
import mlflow
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.agent.workflow import AgentStream

mlflow.llama_index.autolog()
mlflow.set_experiment("llama-index-demo")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

from llama_index.core import set_global_handler

# set_global_handler("simple")

console = Console()


# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a * b


async def main():
    with mlflow.start_run():
        llm = Ollama(model="qwen2.5:7b", verbose=True)

        # we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
        multiply_agent = ReActAgent(
            name="multiply_agent",
            description="Is able to multiply two integers",
            system_prompt="A helpful assistant that can use a tool to multiply numbers.",
            tools=[multiply],
            llm=llm,
            verbose=True,
        )

        addition_agent = ReActAgent(
            name="add_agent",
            description="Is able to add two integers",
            system_prompt="A helpful assistant that can use a tool to add numbers.",
            tools=[add],
            llm=llm,
            verbose=True,
        )

        workflow = AgentWorkflow(
            agents=[multiply_agent, addition_agent],
            root_agent="multiply_agent",
            initial_state={"num_fn_calls": 0},
            state_prompt="Current state: {state}. User message: {msg}",
            verbose=True,
        )

        # run the workflow with context
        ctx = Context(workflow)
        response = await workflow.run(
            user_msg="Can you add 5 and 3 then multiply the result by 2?", ctx=ctx
        )

        # pull out and inspect the state
        state = await ctx.get("state")
        console.print(f"Number of function calls: {state['num_fn_calls']}")
        console.print(f"Response: {response}")


asyncio.run(main())
