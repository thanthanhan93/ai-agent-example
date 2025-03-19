from llama_index.llms.ollama import Ollama
import os
import dotenv
from rich import print
import asyncio
from llama_index.core.llms import ChatMessage, CompletionResponse

dotenv.load_dotenv()

llm = Ollama(model="qwen2.5:7b")


async def stream_llm():
    response = await llm.astream_complete("William Shakespeare is ")

    async for chunk in response:  # Use async for to iterate over the stream
        print(chunk.delta, end="", flush=True)


# asyncio.run(stream_llm())

def chat_llm():
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Tell me a joke."),
    ]
chat_response = llm.chat(messages)
print(chat_response.message)


  